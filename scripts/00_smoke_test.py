"""Run a minimal GPU, no-RAG, and RAG generation smoke test."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_PATH = Path("outputs/smoke_test.jsonl")


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the smoke test; CPU fallback is disabled.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer the question briefly. Return only the answer, with no explanation.",
        },
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return answer.splitlines()[0].strip()


def main() -> None:
    tokenizer, model = load_model()
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model did not load in 4-bit mode.")

    question = "What is the capital of France?"
    no_rag_answer = generate_answer(tokenizer, model, question)
    rag_prompt = (
        "Evidence: Paris is the capital and largest city of France.\n\n"
        f"Question: {question}"
    )
    rag_answer = generate_answer(tokenizer, model, rag_prompt)

    records = [
        {"condition": "no_rag", "question": question, "answer": no_rag_answer},
        {"condition": "rag", "question": question, "answer": rag_answer},
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"no_rag: {no_rag_answer}")
    print(f"rag: {rag_answer}")
    print(f"Saved {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
