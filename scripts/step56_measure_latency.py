"""Measure frozen test-set latency after model load and warm-up."""

from __future__ import annotations

import json
import hashlib
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_from_inputs, prompt_contract_hash
except ImportError:
    from generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_from_inputs, prompt_contract_hash


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nearest_rank_percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    index = max(0, (len(values) * percentile + 99) // 100 - 1)
    return sorted(values)[index]


def summarize_latency(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        condition: {
            metric: {
                "median": statistics.median(values := [row[metric] for row in rows if row["condition"] == condition]),
                "p95": nearest_rank_percentile(values, 95),
            }
            for metric in metrics
        }
        for condition in ("no_rag", "clean_rag", "imperfect_rag")
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    freeze = json.loads((root / "data/manifests/prompt_freeze.json").read_text(encoding="utf-8"))
    if freeze["prompt_hash"] != prompt_contract_hash():
        raise ValueError("runtime prompt does not match the frozen prompt hash")
    questions = [row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "test"]
    reviews = {str(row["question_id"]): row for row in read_jsonl(root / "data/manifests/passage_verification_sheet.jsonl")}
    retrieval_rows = read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")
    retrieval_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in retrieval_rows:
        if row["rank"] <= 3:
            retrieval_by_id.setdefault(str(row["question_id"]), []).append(row)
    for rows in retrieval_by_id.values():
        rows.sort(key=lambda row: row["rank"])

    quantization = config["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(quantization["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["tokenizer_revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["model_revision"], quantization_config=quantization_config,
        device_map="auto", dtype=torch.float16, local_files_only=True,
    )
    model.eval()

    def documents_for(question: dict[str, Any], condition: str) -> tuple[list[str], list[str]]:
        question_id = str(question["question_id"])
        if condition == "no_rag":
            return [], []
        if condition == "clean_rag":
            documents = [reviews[question_id]["clean_doc_1"], reviews[question_id]["clean_doc_2"]]
        else:
            documents = [row["chunk_text"] for row in retrieval_by_id[question_id]]
        ids = [match for document in documents for match in re.findall(r"\[([^\]]+)\]", document)]
        return documents, ids

    warmup_count = 10
    for index in range(warmup_count):
        question = questions[index % len(questions)]
        documents, _ = documents_for(question, "imperfect_rag")
        messages = build_messages(question["question"], "imperfect_rag", documents)
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        generate_from_inputs(tokenizer, model, inputs, max_new_tokens=int(config["max_new_tokens"]), max_input_tokens=int(config["max_input_tokens"]))
    sync_cuda()

    settings = {**GENERATION_SETTINGS, "max_input_tokens": int(config["max_input_tokens"]), "quantization": quantization}
    rows: list[dict[str, Any]] = []
    frozen_predictions_path = root / "outputs/test_predictions.jsonl"
    frozen_predictions_sha256_before = file_sha256(frozen_predictions_path)
    frozen_predictions = {
        (row["question_id"], row["condition"]): row["raw_prediction"]
        for row in read_jsonl(frozen_predictions_path)
    }
    output_path = root / "outputs/test_latency.jsonl"
    if output_path.exists():
        output_path.unlink()
    for condition in ("no_rag", "clean_rag", "imperfect_rag"):
        for question in questions:
            sync_cuda()
            total_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            documents, chunk_ids = documents_for(question, condition)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            prompt_start = time.perf_counter()
            messages = build_messages(question["question"], condition, documents)
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            input_token_count = int(inputs["input_ids"].shape[-1])
            prompt_construction_latency_ms = (time.perf_counter() - prompt_start) * 1000

            sync_cuda()
            generation_start = time.perf_counter()
            raw_prediction = generate_from_inputs(tokenizer, model, inputs, max_new_tokens=int(config["max_new_tokens"]), max_input_tokens=int(config["max_input_tokens"]))
            sync_cuda()
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000
            sync_cuda()
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            row = {
                "question_id": question["question_id"], "condition": condition,
                "prompt_hash": freeze["prompt_hash"], "input_token_count": input_token_count,
                "output_token_count": len(tokenizer(raw_prediction, add_special_tokens=False)["input_ids"]),
                "raw_prediction": raw_prediction, "retrieved_chunk_ids": chunk_ids,
                "retrieval_result_lookup_latency_ms": round(retrieval_latency_ms, 3),
                "retrieval_source": "frozen_result_lookup",
                "prompt_construction_latency_ms": round(prompt_construction_latency_ms, 3),
                "generation_latency_ms": round(generation_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "model_id": config["model"], "model_revision": str(config["model_revision"]),
                "generation_settings": settings,
            }
            append_jsonl(output_path, row)
            rows.append(row)

    frozen_predictions_sha256_after = file_sha256(frozen_predictions_path)
    timing_predictions = {(row["question_id"], row["condition"]): row["raw_prediction"] for row in rows}
    answers_identical = timing_predictions == frozen_predictions
    summary = {
        "rows": len(rows),
        "warmup_generations": warmup_count,
        "model_loading_included": False,
        "retrieval_metric": "retrieval-result lookup latency",
        "answers_identical_to_frozen_predictions": answers_identical,
        "frozen_predictions_sha256_before": frozen_predictions_sha256_before,
        "frozen_predictions_sha256_after": frozen_predictions_sha256_after,
        "metrics_ms": summarize_latency(
            rows,
            ("retrieval_result_lookup_latency_ms", "prompt_construction_latency_ms", "generation_latency_ms", "total_latency_ms"),
        ),
    }
    summary_path = root / "outputs/test_latency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "latency_artifact": "outputs/test_latency.jsonl",
        "latency_artifact_sha256": file_sha256(output_path),
        "summary_artifact": "outputs/test_latency_summary.json",
        "summary_artifact_sha256": file_sha256(summary_path),
        "row_count": len(rows),
        "warmup_generations": warmup_count,
        "model_loading_included": False,
        "answers_identical_to_frozen_predictions": answers_identical,
        "frozen_predictions_sha256_before": frozen_predictions_sha256_before,
        "frozen_predictions_sha256_after": frozen_predictions_sha256_after,
    }
    (root / "outputs/test_latency.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
