"""Validate all three generation conditions on the frozen development split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_continuation, looks_explanatory, parse_answer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    model_id = config["model"]
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    max_input_tokens = int(config["max_input_tokens"])
    max_new_tokens = int(config["max_new_tokens"])
    questions = [row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "dev"]
    reviews = {str(row["question_id"]): row for row in read_jsonl(root / "data/manifests/passage_verification_sheet.jsonl")}
    retrieval_rows = read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")

    quantization = config["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(quantization["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    context_limit = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))

    records: list[dict] = []
    for question in questions:
        question_id = str(question["question_id"])
        review = reviews[question_id]
        top_three = sorted(
            (row for row in retrieval_rows if str(row["question_id"]) == question_id and row["rank"] <= 3),
            key=lambda row: row["rank"],
        )
        conditions = {
            "no_rag": [],
            "clean_rag": [review["clean_doc_1"], review["clean_doc_2"]],
            "imperfect_rag": [row["chunk_text"] for row in top_three],
        }
        for condition, documents in conditions.items():
            prompt = build_prompt(question["question"], condition, documents)
            messages = build_messages(question["question"], condition, documents)
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            record = {
                "question_id": question["question_id"],
                "condition": condition,
                "question": question["question"],
                "document_count": len(documents),
                "prompt_tokens": prompt_tokens,
                "max_input_tokens": max_input_tokens,
                "context_limit": context_limit,
                "documents_truncated": False,
                "generation_settings": GENERATION_SETTINGS,
                "model": model_id,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
            }
            if prompt_tokens + max_new_tokens > max_input_tokens:
                record.update({"status": "overflow_rejected", "error": "prompt plus generation exceeds max_input_tokens"})
                records.append(record)
                continue

            raw = generate_continuation(
                tokenizer,
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=bool(config["do_sample"]),
                max_input_tokens=max_input_tokens,
            )
            parsed = parse_answer(raw)
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            words = parsed.split()
            record.update({
                "status": "ok",
                "raw_continuation": raw,
                "parsed_answer": parsed,
                "raw_empty": not bool(raw),
                "parsed_empty": not bool(parsed),
                "short_answer": len(words) <= 8,
                "word_count": len(words),
                "has_explanation": looks_explanatory(raw),
                "parser_correct": parsed == (lines[0] if lines else ""),
            })
            records.append(record)

    summary = {
        "question_count": len(questions),
        "expected_records": len(questions) * 3,
        "actual_records": len(records),
        "status_counts": {status: sum(row["status"] == status for row in records) for status in {row["status"] for row in records}},
        "empty_raw": sum(row.get("raw_empty", False) for row in records),
        "empty_parsed": sum(row.get("parsed_empty", False) for row in records),
        "short_answers": sum(row.get("short_answer", False) for row in records),
        "explanations": sum(row.get("has_explanation", False) for row in records),
        "documents_truncated": sum(row["documents_truncated"] for row in records),
        "parser_correct": sum(row.get("parser_correct", False) for row in records),
        "prompt_overflow_rejected": sum(row["status"] == "overflow_rejected" for row in records),
        "model": model_id,
        "model_revision": model_revision,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(config["do_sample"]),
    }
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step53_dev_validation.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "step53_dev_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
