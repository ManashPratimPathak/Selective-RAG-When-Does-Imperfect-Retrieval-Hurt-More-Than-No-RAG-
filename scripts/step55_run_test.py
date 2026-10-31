"""Run the frozen prompt on the test split and checkpoint every prediction."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_continuation, prompt_contract_hash
except ImportError:
    from generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_continuation, prompt_contract_hash


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prediction_row(**fields: Any) -> dict[str, Any]:
    return {
        "question_id": fields["question_id"],
        "condition": fields["condition"],
        "prompt_hash": fields["prompt_hash"],
        "input_token_count": fields["input_token_count"],
        "output_token_count": fields["output_token_count"],
        "raw_prediction": fields["raw_prediction"],
        "latency_ms": fields["latency_ms"],
        "retrieved_chunk_ids": fields["retrieved_chunk_ids"],
        "model_id": fields["model_id"],
        "model_revision": fields["model_revision"],
        "generation_settings": fields["generation_settings"],
    }


def validate_resume_rows(
    rows: list[dict[str, Any]],
    question_ids: list[int],
    prompt_hash: str,
    model_id: str,
    model_revision: str,
) -> set[tuple[int, str]]:
    expected_ids = set(question_ids)
    conditions = {"no_rag", "clean_rag", "imperfect_rag"}
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if row.get("prompt_hash") != prompt_hash:
            raise ValueError("resume row has a different prompt hash")
        if row.get("model_revision") != model_revision:
            raise ValueError("resume row has a different model revision")
        if row.get("model_id") != model_id:
            raise ValueError("resume row has a different model ID")
        key = (row["question_id"], row["condition"])
        if key[0] not in expected_ids or key[1] not in conditions:
            raise ValueError(f"resume row is outside the frozen test set: {key}")
        if key in seen:
            raise ValueError(f"duplicate resume row: {key}")
        seen.add(key)
    return seen


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def _document_ids(documents: list[str]) -> list[str]:
    return [match for document in documents for match in re.findall(r"\[([^\]]+)\]", document)]


def _append_row(path: Path, row: dict[str, Any]) -> None:
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
    for setting in ("do_sample", "max_new_tokens", "num_beams", "use_cache", "batch_size"):
        if config[setting] != GENERATION_SETTINGS[setting]:
            raise ValueError(f"config disagrees with generation contract for {setting}")

    questions = [row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "test"]
    if len(questions) != int(config["test_size"]):
        raise ValueError(f"expected {config['test_size']} test questions, found {len(questions)}")
    reviews = {str(row["question_id"]): row for row in read_jsonl(root / "data/manifests/passage_verification_sheet.jsonl")}
    retrieval_rows = read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")

    output_path = root / "outputs/test_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    completed = validate_resume_rows(
        existing_rows,
        [row["question_id"] for row in questions],
        freeze["prompt_hash"],
        config["model"],
        str(config["model_revision"]),
    )
    if len(completed) == len(questions) * 3:
        digest = artifact_sha256(output_path)
        manifest = {
            "artifact": "outputs/test_predictions.jsonl",
            "sha256": digest,
            "row_count": len(existing_rows),
            "question_count": len(questions),
            "conditions": ["no_rag", "clean_rag", "imperfect_rag"],
            "prompt_hash": freeze["prompt_hash"],
            "model_id": config["model"],
            "model_revision": str(config["model_revision"]),
        }
        (root / "outputs/test_predictions.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        return

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

    generation_settings = {
        **GENERATION_SETTINGS,
        "max_input_tokens": int(config["max_input_tokens"]),
        "quantization": quantization,
    }
    conditions = ("no_rag", "clean_rag", "imperfect_rag")
    for condition in conditions:
        for question in questions:
            key = (question["question_id"], condition)
            if key in completed:
                continue
            question_id = str(question["question_id"])
            review = reviews[question_id]
            top_three = sorted(
                (row for row in retrieval_rows if str(row["question_id"]) == question_id and row["rank"] <= 3),
                key=lambda row: row["rank"],
            )
            documents = {
                "no_rag": [],
                "clean_rag": [review["clean_doc_1"], review["clean_doc_2"]],
                "imperfect_rag": [row["chunk_text"] for row in top_three],
            }[condition]
            prompt = build_prompt(question["question"], condition, documents)
            messages = build_messages(question["question"], condition, documents)
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_token_count = int(encoded["input_ids"].shape[-1])
            max_input_tokens = int(config["max_input_tokens"])
            if input_token_count + int(config["max_new_tokens"]) > max_input_tokens:
                raise ValueError(f"input exceeds frozen maximum for question {question['question_id']}")

            started = time.perf_counter()
            raw_prediction = generate_continuation(
                tokenizer,
                model,
                prompt,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=bool(config["do_sample"]),
                max_input_tokens=max_input_tokens,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            row = build_prediction_row(
                question_id=question["question_id"],
                condition=condition,
                prompt_hash=freeze["prompt_hash"],
                input_token_count=input_token_count,
                output_token_count=_token_count(tokenizer, raw_prediction),
                raw_prediction=raw_prediction,
                latency_ms=latency_ms,
                retrieved_chunk_ids=_document_ids(documents),
                model_id=config["model"],
                model_revision=str(config["model_revision"]),
                generation_settings=generation_settings,
            )
            _append_row(output_path, row)
            completed.add(key)
            print(f"{condition}: {len(completed)}/600 question_id={question['question_id']}")

    final_rows = read_jsonl(output_path)
    validate_resume_rows(
        final_rows,
        [row["question_id"] for row in questions],
        freeze["prompt_hash"],
        config["model"],
        str(config["model_revision"]),
    )
    expected_count = len(questions) * 3
    if len(final_rows) != expected_count:
        raise ValueError(f"expected {expected_count} final rows, found {len(final_rows)}")
    digest = artifact_sha256(output_path)
    manifest = {
        "artifact": "outputs/test_predictions.jsonl",
        "sha256": digest,
        "row_count": len(final_rows),
        "question_count": len(questions),
        "conditions": list(conditions),
        "prompt_hash": freeze["prompt_hash"],
        "model_id": config["model"],
        "model_revision": str(config["model_revision"]),
    }
    (root / "outputs/test_predictions.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
