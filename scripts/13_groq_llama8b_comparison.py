"""Llama-3.1-8B-Instant comparison via Groq API.

Evaluates Llama-3.1-8B against frozen Qwen2.5-1.5B results on:
1. no_rag: Baseline parametric memory (Head vs Tail)
2. imperfect_rag: Realistic BM25 top-3 retrieval (Rescue vs Damage)
3. temporal_rag: Adversarial temporal distractor stress test

Outputs:
- outputs/llama8b_comparison/predictions.jsonl
- outputs/llama8b_comparison/summary.json
- outputs/llama8b_comparison/comparison_table.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests

try:
    from .answer_normalization import normalize_answer, normalize_answers
except ImportError:
    from answer_normalization import normalize_answer, normalize_answers


SYSTEM_INSTRUCTION = (
    "Answer the factual question using the supplied documents when they are relevant. "
    "Ignore documents that do not help. Return only the shortest answer phrase, without explanation."
)
USER_PROMPT_TEMPLATE = "Documents:\n{documents}\n\nQuestion:\n{question}\n\nShortest answer:"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


def build_prompt(question: str, condition: str, documents: Sequence[str]) -> str:
    if condition == "no_rag":
        document_block = "No documents provided."
    else:
        document_block = "\n".join(
            f"[D{index}] {document}" for index, document in enumerate(documents, start=1)
        )
    return USER_PROMPT_TEMPLATE.format(documents=document_block, question=question)


def build_messages(question: str, condition: str, documents: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_prompt(question, condition, documents)},
    ]


def parse_answer(continuation: str) -> str:
    text = continuation.strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def alias_aware_exact_match(norm_pred: str, norm_aliases: list[str]) -> bool:
    return bool(norm_pred) and any(norm_pred == a for a in norm_aliases if a)


def token_f1(norm_pred: str, norm_alias: str) -> float:
    pred_tokens = norm_pred.split()
    alias_tokens = norm_alias.split()
    if not pred_tokens or not alias_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(alias_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(alias_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def max_token_f1(norm_pred: str, norm_aliases: list[str]) -> float:
    return max((token_f1(norm_pred, a) for a in norm_aliases), default=0.0)


def call_groq_api(
    messages: list[dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> tuple[str, float]:
    """Call Groq API with robust exponential backoff on rate limits."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }

    wait_time = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.perf_counter()
            resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"] or ""
                return raw_text, latency_ms

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                sleep_dur = float(retry_after) if retry_after else wait_time
                print(f" [Rate limit 429. Backing off for {sleep_dur:.1f}s (attempt {attempt}/{max_retries})...]", flush=True)
                time.sleep(sleep_dur)
                wait_time *= backoff_factor
                continue

            print(f" [API error {resp.status_code}: {resp.text}. Retrying in {wait_time:.1f}s...]", flush=True)
            time.sleep(wait_time)
            wait_time *= backoff_factor

        except requests.RequestException as e:
            print(f" [Network exception: {e}. Retrying in {wait_time:.1f}s...]", flush=True)
            time.sleep(wait_time)
            wait_time *= backoff_factor

    raise RuntimeError(f"Groq API call failed after {max_retries} retries.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scale comparison on Groq API.")
    parser.add_argument("--api-key", default=None, help="Groq API key (defaults to GROQ_API_KEY env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on test questions (for smoke tests)")
    parser.add_argument("--delay", type=float, default=2.1, help="Delay in seconds between requests (default 2.1s for 30 RPM limit)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate metrics on existing outputs without API calls")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not args.dry_run and not api_key:
        print("\n[ERROR] GROQ_API_KEY is not set!")
        print("Please provide it via --api-key <key> or set $env:GROQ_API_KEY=\"<key>\" in PowerShell.")
        print("You can get a free key at: https://console.groq.com/keys\n")
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    retrieval_path = root / "data/retrieval/retrieval_results_top5.jsonl"
    impurities_path = root / "data/impurities/adversarial_passages.jsonl"

    output_dir = root / "outputs/llama8b_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "comparison_table.md"

    # Load frozen test questions
    questions = [q for q in read_jsonl(sample_path) if q.get("split") == "test"]
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} test questions.")

    # Load BM25 retrieval results (top 3)
    bm25_by_qid: dict[int, list[str]] = {}
    for row in read_jsonl(retrieval_path):
        qid = row["question_id"]
        if row.get("rank", 99) <= 3:
            bm25_by_qid.setdefault(qid, []).append(row["chunk_text"])

    # Load temporal distractors
    temporal_by_qid = {row["question_id"]: row for row in read_jsonl(impurities_path)}

    # Check already completed predictions for resuming
    existing_preds = read_jsonl(pred_path)
    completed = {(p["question_id"], p["condition"]) for p in existing_preds}
    print(f"Found {len(completed)} existing cached predictions.")

    conditions = ["no_rag", "imperfect_rag", "temporal_rag"]

    # Run generations
    if not args.dry_run:
        total_to_run = sum(
            1 for c in conditions for q in questions if (q["question_id"], c) not in completed
        )
        print(f"Total generations to run: {total_to_run}")

        count = 0
        for cond in conditions:
            print(f"\n==========================================")
            print(f"  Running Condition: {cond}")
            print(f"==========================================")
            for q in questions:
                qid = q["question_id"]
                if (qid, cond) in completed:
                    continue

                # Prepare documents for condition
                if cond == "no_rag":
                    docs = []
                elif cond == "imperfect_rag":
                    docs = bm25_by_qid.get(qid, [])[:3]
                elif cond == "temporal_rag":
                    t_row = temporal_by_qid.get(qid, {})
                    docs = [
                        t_row.get("clean_gold_doc", ""),
                        t_row.get("temporal_doc_1", ""),
                        t_row.get("temporal_doc_2", ""),
                    ]
                else:
                    docs = []

                messages = build_messages(q["question"], cond, docs)
                raw_pred, latency_ms = call_groq_api(messages, api_key=api_key, model=args.model)
                parsed_pred = parse_answer(raw_pred)

                norm_pred = normalize_answer(parsed_pred)
                norm_aliases = normalize_answers(q["accepted_answers"])
                is_em = alias_aware_exact_match(norm_pred, norm_aliases)
                f1 = max_token_f1(norm_pred, norm_aliases)

                record = {
                    "question_id": qid,
                    "model_id": args.model,
                    "condition": cond,
                    "popularity_group": q.get("popularity_group", "unknown"),
                    "relation": q.get("relation", ""),
                    "raw_prediction": raw_pred,
                    "parsed_prediction": parsed_pred,
                    "normalized_prediction": norm_pred,
                    "accepted_answers": q["accepted_answers"],
                    "exact_match": is_em,
                    "token_f1": round(f1 * 100, 2),
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(pred_path, record)
                completed.add((qid, cond))
                count += 1

                print(f"[{count}/{total_to_run}] QID {qid} ({cond}) -> '{parsed_pred}' (EM: {is_em})", flush=True)
                time.sleep(args.delay)

    # Scoring and Aggregate Analysis
    all_preds = read_jsonl(pred_path)
    if not all_preds:
        print("[WARNING] No predictions found to evaluate.")
        return

    # Index by condition and question_id
    preds_by_cond: dict[str, dict[int, dict[str, Any]]] = {}
    for p in all_preds:
        preds_by_cond.setdefault(p["condition"], {})[p["question_id"]] = p

    # Baseline no-rag answers
    no_rag_map = preds_by_cond.get("no_rag", {})

    metrics_summary: dict[str, Any] = {}
    for cond in conditions:
        c_preds = preds_by_cond.get(cond, {})
        if not c_preds:
            continue

        n_total = len(c_preds)
        em_count = sum(1 for p in c_preds.values() if p["exact_match"])
        em_pct = round(em_count / n_total * 100, 2)
        avg_f1 = round(sum(p["token_f1"] for p in c_preds.values()) / n_total, 2)

        # Head vs Tail
        head_preds = [p for p in c_preds.values() if p.get("popularity_group") == "head"]
        tail_preds = [p for p in c_preds.values() if p.get("popularity_group") == "tail"]

        head_em = round(sum(1 for p in head_preds if p["exact_match"]) / len(head_preds) * 100, 2) if head_preds else 0.0
        tail_em = round(sum(1 for p in tail_preds if p["exact_match"]) / len(tail_preds) * 100, 2) if tail_preds else 0.0

        # Transitions relative to No-RAG
        damage_count = 0
        rescue_count = 0
        no_rag_correct_count = 0
        no_rag_incorrect_count = 0

        for qid, p in c_preds.items():
            if qid in no_rag_map:
                nr_correct = no_rag_map[qid]["exact_match"]
                c_correct = p["exact_match"]
                if nr_correct:
                    no_rag_correct_count += 1
                    if not c_correct:
                        damage_count += 1
                else:
                    no_rag_incorrect_count += 1
                    if c_correct:
                        rescue_count += 1

        damage_rate = round(damage_count / no_rag_correct_count * 100, 2) if no_rag_correct_count else 0.0
        rescue_rate = round(rescue_count / no_rag_incorrect_count * 100, 2) if no_rag_incorrect_count else 0.0

        metrics_summary[cond] = {
            "n": n_total,
            "em_percent": em_pct,
            "token_f1_percent": avg_f1,
            "head_em_percent": head_em,
            "tail_em_percent": tail_em,
            "damage_count": damage_count,
            "damage_rate_percent": damage_rate,
            "rescue_count": rescue_count,
            "rescue_rate_percent": rescue_rate,
        }

    # Save summary
    summary_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    print(f"\nSaved metrics summary to: {summary_path}")

    # Build Side-by-Side Comparison Markdown
    # Qwen baseline numbers
    qwen = {
        "no_rag": {"em": 13.0, "f1": 18.08, "head_em": 21.0, "tail_em": 5.0, "damage": "0.0% (0)", "rescue": "0.0% (0)"},
        "imperfect_rag": {"em": 61.5, "f1": 71.77, "head_em": 55.0, "tail_em": 68.0, "damage": "23.08% (6)", "rescue": "59.20% (103)"},
        "temporal_rag": {"em": 57.0, "f1": 67.89, "head_em": "N/A", "tail_em": "N/A", "damage": "30.77% (8)", "rescue": "55.17% (96)"},
    }

    llama_nr = metrics_summary.get("no_rag", {})
    llama_imp = metrics_summary.get("imperfect_rag", {})
    llama_temp = metrics_summary.get("temporal_rag", {})

    table_md = f"""# Cross-Model Scale Comparison: Qwen2.5-1.5B vs. Llama-3.1-8B

| Metric / Evaluation Tier | Qwen2.5-1.5B (Small) | Llama-3.1-8B (Medium) | Scale Impact (Delta) |
| :--- | :---: | :---: | :---: |
| **No-RAG Baseline (Overall EM)** | {qwen['no_rag']['em']}% | {llama_nr.get('em_percent', 'N/A')}% | {round(llama_nr.get('em_percent', 0) - qwen['no_rag']['em'], 2) if llama_nr else 'N/A'} pp |
| • *Head Entity EM* | {qwen['no_rag']['head_em']}% | {llama_nr.get('head_em_percent', 'N/A')}% | {round(llama_nr.get('head_em_percent', 0) - qwen['no_rag']['head_em'], 2) if llama_nr else 'N/A'} pp |
| • *Tail Entity EM* | {qwen['no_rag']['tail_em']}% | {llama_nr.get('tail_em_percent', 'N/A')}% | {round(llama_nr.get('tail_em_percent', 0) - qwen['no_rag']['tail_em'], 2) if llama_nr else 'N/A'} pp |
| • *Token F1* | {qwen['no_rag']['f1']}% | {llama_nr.get('token_f1_percent', 'N/A')}% | {round(llama_nr.get('token_f1_percent', 0) - qwen['no_rag']['f1'], 2) if llama_nr else 'N/A'} pp |
| **Imperfect RAG (BM25 Top-3 EM)** | {qwen['imperfect_rag']['em']}% | {llama_imp.get('em_percent', 'N/A')}% | {round(llama_imp.get('em_percent', 0) - qwen['imperfect_rag']['em'], 2) if llama_imp else 'N/A'} pp |
| • *Head Entity EM* | {qwen['imperfect_rag']['head_em']}% | {llama_imp.get('head_em_percent', 'N/A')}% | {round(llama_imp.get('head_em_percent', 0) - qwen['imperfect_rag']['head_em'], 2) if llama_imp else 'N/A'} pp |
| • *Tail Entity EM* | {qwen['imperfect_rag']['tail_em']}% | {llama_imp.get('tail_em_percent', 'N/A')}% | {round(llama_imp.get('tail_em_percent', 0) - qwen['imperfect_rag']['tail_em'], 2) if llama_imp else 'N/A'} pp |
| • *Conditional Damage Rate* | {qwen['imperfect_rag']['damage']} | {llama_imp.get('damage_rate_percent', 'N/A')}% ({llama_imp.get('damage_count', 0)}) | - |
| • *Conditional Rescue Rate* | {qwen['imperfect_rag']['rescue']} | {llama_imp.get('rescue_rate_percent', 'N/A')}% ({llama_imp.get('rescue_count', 0)}) | - |
| **Temporal Stress RAG (Adversarial)** | {qwen['temporal_rag']['em']}% | {llama_temp.get('em_percent', 'N/A')}% | {round(llama_temp.get('em_percent', 0) - qwen['temporal_rag']['em'], 2) if llama_temp else 'N/A'} pp |
| • *Conditional Damage Rate* | {qwen['temporal_rag']['damage']} | {llama_temp.get('damage_rate_percent', 'N/A')}% ({llama_temp.get('damage_count', 0)}) | - |
| • *Conditional Rescue Rate* | {qwen['temporal_rag']['rescue']} | {llama_temp.get('rescue_rate_percent', 'N/A')}% ({llama_temp.get('rescue_count', 0)}) | - |

*Artifact provenance: Qwen results from frozen test ledgers; Llama-3.1-8B evaluated via Groq API.*
"""

    table_path.write_text(table_md, encoding="utf-8")
    print(f"Saved comparison table to: {table_path}\n")
    print(table_md)


if __name__ == "__main__":
    main()
