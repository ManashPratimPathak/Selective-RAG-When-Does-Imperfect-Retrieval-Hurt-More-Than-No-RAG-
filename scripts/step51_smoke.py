"""Run one frozen development question through all Step 5.1 conditions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate import GENERATION_SETTINGS, build_messages, build_prompt, generate_answer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    model_id = config["model"]
    for setting in ("do_sample", "max_new_tokens", "num_beams", "use_cache", "batch_size"):
        if config[setting] != GENERATION_SETTINGS[setting]:
            raise ValueError(f"config disagrees with generation contract for {setting}")
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
    if not (snapshot_root / model_revision).is_dir():
        raise FileNotFoundError(f"pinned model revision is not cached: {model_revision}")

    questions = [row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "dev"]
    question = questions[0]
    question_id = str(question["question_id"])
    reviews = {str(row["question_id"]): row for row in read_jsonl(root / "data/manifests/passage_verification_sheet.jsonl")}
    retrieval = [
        row
        for row in read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")
        if str(row["question_id"]) == question_id and row["rank"] <= 3
    ]
    review = reviews[question_id]
    clean_documents = [review["clean_doc_1"], review["clean_doc_2"]]
    imperfect_documents = [row["chunk_text"] for row in sorted(retrieval, key=lambda row: row["rank"])]
    assert review["status"] == "verified_manual"
    assert len(retrieval) == 3

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
    max_input_tokens = int(config["max_input_tokens"])
    if max_input_tokens + int(config["max_new_tokens"]) > context_limit:
        raise ValueError("configured input and generation lengths exceed the model context limit")
    conditions = {
        "no_rag": [],
        "clean_rag": clean_documents,
        "imperfect_rag": imperfect_documents,
    }
    records = []
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
        answer = generate_answer(
            tokenizer,
            model,
            prompt,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=bool(config["do_sample"]),
            max_input_tokens=max_input_tokens,
        )
        records.append({
            "condition": condition,
            "question_id": question["question_id"],
            "question": question["question"],
            "document_count": len(documents),
            "retrieval_source": {
                "no_rag": "none",
                "clean_rag": "data/manifests/passage_verification_sheet.jsonl:verified_manual:clean_doc_1,clean_doc_2",
                "imperfect_rag": "data/retrieval/retrieval_results_top5.jsonl:rank<=3",
            }[condition],
            "answer": answer,
            "parseable": bool(answer) and "\n" not in answer,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": prompt_tokens,
            "max_input_tokens": max_input_tokens,
            "context_limit": context_limit,
            "max_new_tokens": int(config["max_new_tokens"]),
            "do_sample": bool(config["do_sample"]),
            "num_beams": int(config["num_beams"]),
            "use_cache": bool(config["use_cache"]),
            "batch_size": int(config["batch_size"]),
            "quantization": quantization,
            "model": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
        })

    output = root / "outputs/step51_smoke.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "question_id": question["question_id"], "records": records}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
