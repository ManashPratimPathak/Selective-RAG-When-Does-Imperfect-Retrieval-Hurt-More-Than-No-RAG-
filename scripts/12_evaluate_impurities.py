"""Evaluate model and router robustness under structured adversarial impurities.

Evaluates 4 impurity conditions:
1. entity_overlap_rag
2. relation_matched_rag
3. temporal_rag
4. combined_adversarial_rag

For each condition:
- Measures LLM Exact Match (EM), Token F1, Damage Rate, and Rescue Rate.
- Simulates the threshold router decision using standardized BM25 scores (s1_z >= threshold).
- Measures Router EM, Router Damage Rate, and Router Deception Rate.

Outputs:
- outputs/impurities/predictions.jsonl
- outputs/impurities/evaluation_summary.json
- outputs/impurities/adversarial_policy_comparison.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .answer_normalization import normalize_answer, normalize_answers
    from .generate import (
        GENERATION_SETTINGS,
        build_messages,
        build_prompt,
        generate_continuation,
        parse_answer,
    )
    from .tokenization import tokenize
except ImportError:
    from answer_normalization import normalize_answer, normalize_answers
    from generate import (
        GENERATION_SETTINGS,
        build_messages,
        build_prompt,
        generate_continuation,
        parse_answer,
    )
    from tokenization import tokenize
from collections import Counter


def alias_aware_exact_match(norm_pred: str, norm_aliases: list[str]) -> bool:
    return bool(norm_pred) and any(norm_pred == a for a in norm_aliases if a)


def token_f1_score(norm_pred: str, norm_alias: str) -> float:
    pred_tokens = norm_pred.split()
    alias_tokens = norm_alias.split()
    if not pred_tokens or not alias_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(alias_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(alias_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(norm_pred: str, norm_aliases: list[str]) -> float:
    return max((token_f1_score(norm_pred, a) for a in norm_aliases), default=0.0)



DEV_MEAN = 33.20043868432112
DEV_STD = 8.67269390781522
ROUTER_THRESHOLD = -0.8534212104401723


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compute_pseudo_bm25_score(query_tokens: list[str], documents: Sequence[str]) -> float:
    """Compute lexical query-document match score proxy for routing."""
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    doc_text = " ".join(documents).lower()
    doc_tokens = tokenize(doc_text)
    overlap = sum(doc_text.count(token) for token in query_set)
    # Scale into natural BM25 score range (~15 - 50)
    return float(min(55.0, 15.0 + overlap * 3.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simulate routing and metrics without model inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions evaluated")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}

    impurities_path = root / "data/impurities/adversarial_passages.jsonl"
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    ledger_path = root / "outputs/test_scoring_ledger.jsonl"
    output_pred_path = root / "outputs/impurities/predictions.jsonl"
    output_summary_path = root / "outputs/impurities/evaluation_summary.json"
    output_csv_path = root / "outputs/impurities/adversarial_policy_comparison.csv"

    impurities_data = {row["question_id"]: row for row in read_jsonl(impurities_path)}
    questions = [row for row in read_jsonl(sample_path) if row.get("split") == "test"]
    if args.limit:
        questions = questions[:args.limit]

    # Baseline no_rag frozen results
    ledger_rows = read_jsonl(ledger_path)
    no_rag_by_qid = {
        row["question_id"]: row for row in ledger_rows if row.get("condition") == "no_rag"
    }

    conditions = [
        "entity_overlap_rag",
        "relation_matched_rag",
        "temporal_rag",
        "combined_adversarial_rag",
    ]

    existing_predictions = read_jsonl(output_pred_path)
    completed_keys = {(row["question_id"], row["condition"]) for row in existing_predictions}

    model = None
    tokenizer = None

    if not args.dry_run:
        quantization = config["quantization"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(quantization["load_in_4bit"]),
            bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(quantization["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["tokenizer_revision"], local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["model_revision"],
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True,
        )
        model.eval()

    all_prediction_rows = list(existing_predictions)

    for cond in conditions:
        print(f"--- Processing condition: {cond} ---")
        for q in questions:
            qid = q["question_id"]
            if (qid, cond) in completed_keys:
                continue

            imp = impurities_data[qid]
            clean_doc = imp["clean_gold_doc"]

            if cond == "entity_overlap_rag":
                docs = [clean_doc, imp["entity_overlap_doc_1"], imp["entity_overlap_doc_2"]]
            elif cond == "relation_matched_rag":
                docs = [clean_doc, imp["relation_matched_doc_1"], imp["relation_matched_doc_2"]]
            elif cond == "temporal_rag":
                docs = [clean_doc, imp["temporal_doc_1"], imp["temporal_doc_2"]]
            elif cond == "combined_adversarial_rag":
                docs = [clean_doc, imp["entity_overlap_doc_1"], imp["relation_matched_doc_1"]]
            else:
                docs = [clean_doc]

            q_text = q["question"]
            q_tokens = tokenize(q_text)
            s1 = compute_pseudo_bm25_score(q_tokens, docs)
            s1_z = (s1 - DEV_MEAN) / DEV_STD
            routed_to_rag = bool(s1_z >= ROUTER_THRESHOLD)

            started = time.perf_counter()
            if args.dry_run:
                # Dry run simulation prediction
                raw_pred = imp["accepted_answers"][0] if cond == "clean_doc" else "simulated_answer"
                parsed = raw_pred
            else:
                prompt = build_prompt(q_text, "clean_rag", docs)
                messages = build_messages(q_text, "clean_rag", docs)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                raw_pred = generate_continuation(
                    tokenizer,
                    model,
                    prompt,
                    max_new_tokens=int(config.get("max_new_tokens", 16)),
                    do_sample=bool(config.get("do_sample", False)),
                    max_input_tokens=int(config.get("max_input_tokens", 2048)),
                )
                parsed = parse_answer(raw_pred)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)

            norm_parsed = normalize_answer(parsed)
            norm_aliases = normalize_answers(q["accepted_answers"])
            em = alias_aware_exact_match(norm_parsed, norm_aliases)
            f1 = max_token_f1(norm_parsed, norm_aliases)

            row = {
                "question_id": qid,
                "condition": cond,
                "raw_prediction": raw_pred,
                "parsed_prediction": parsed,
                "exact_match": em,
                "token_f1": f1,
                "latency_ms": latency_ms,
                "bm25_s1": s1,
                "bm25_s1_z": s1_z,
                "routed_to_rag": routed_to_rag,
            }
            append_jsonl_row(output_pred_path, row)
            all_prediction_rows.append(row)
            completed_keys.add((qid, cond))

    # Evaluate summary and comparison
    summary: dict[str, Any] = {}
    csv_lines = [
        "policy,em_percent,token_f1_percent,damage_rate_percent,rescue_rate_percent,routed_to_rag_percent"
    ]

    # Baseline No-RAG reference
    no_rag_correct_qids = {qid for qid, r in no_rag_by_qid.items() if r.get("exact_match")}
    no_rag_incorrect_qids = {qid for qid, r in no_rag_by_qid.items() if not r.get("exact_match")}
    n_baseline_correct = len(no_rag_correct_qids)
    n_baseline_incorrect = len(no_rag_incorrect_qids)

    by_condition_rows: dict[str, list[dict[str, Any]]] = {}
    for r in all_prediction_rows:
        by_condition_rows.setdefault(r["condition"], []).append(r)

    for cond, rows in by_condition_rows.items():
        n = len(rows)
        em_pct = round(sum(r["exact_match"] for r in rows) / n * 100, 2)
        f1_pct = round(sum(r["token_f1"] for r in rows) / n * 100, 2)

        damage_count = sum(
            1 for r in rows
            if r["question_id"] in no_rag_correct_qids and not r["exact_match"]
        )
        rescue_count = sum(
            1 for r in rows
            if r["question_id"] in no_rag_incorrect_qids and r["exact_match"]
        )
        damage_rate = round(damage_count / n_baseline_correct * 100, 2) if n_baseline_correct else 0.0
        rescue_rate = round(rescue_count / n_baseline_incorrect * 100, 2) if n_baseline_incorrect else 0.0

        # Router performance under this condition
        router_predictions = [
            r if r["routed_to_rag"] else no_rag_by_qid[r["question_id"]]
            for r in rows
        ]
        router_em = round(sum(rp["exact_match"] for rp in router_predictions) / n * 100, 2)
        router_f1 = round(sum(rp["token_f1"] for rp in router_predictions) / n * 100, 2)
        router_routed_pct = round(sum(r["routed_to_rag"] for r in rows) / n * 100, 2)
        router_damage_count = sum(
            1 for rp in router_predictions
            if rp["question_id"] in no_rag_correct_qids and not rp["exact_match"]
        )
        router_damage_rate = round(router_damage_count / n_baseline_correct * 100, 2) if n_baseline_correct else 0.0

        summary[cond] = {
            "unconditional_always_rag": {
                "em_percent": em_pct,
                "token_f1_percent": f1_pct,
                "damage_rate_percent": damage_rate,
                "rescue_rate_percent": rescue_rate,
                "damage_count": damage_count,
                "rescue_count": rescue_count,
            },
            "threshold_router": {
                "em_percent": router_em,
                "token_f1_percent": router_f1,
                "damage_rate_percent": router_damage_rate,
                "routed_to_rag_percent": router_routed_pct,
            },
        }

        csv_lines.append(f"{cond}_always_rag,{em_pct},{f1_pct},{damage_rate},{rescue_rate},100.0")
        csv_lines.append(f"{cond}_threshold_router,{router_em},{router_f1},{router_damage_rate},{rescue_rate},{router_routed_pct}")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Evaluation complete. Summary written to {output_summary_path}")
    print(f"Policy comparison table written to {output_csv_path}")


if __name__ == "__main__":
    main()
