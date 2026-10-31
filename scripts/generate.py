"""Matched prompts and Qwen generation helpers for the three RAG conditions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

import torch


SYSTEM_INSTRUCTION = (
    "Answer the factual question using the supplied documents when they are relevant. "
    "Ignore documents that do not help. Return only the shortest answer phrase, without explanation."
)
USER_PROMPT_TEMPLATE = "Documents:\n{documents}\n\nQuestion:\n{question}\n\nShortest answer:"
CONDITIONS = {"no_rag", "clean_rag", "imperfect_rag"}
GENERATION_SETTINGS = {
    "do_sample": False,
    "max_new_tokens": 16,
    "num_beams": 1,
    "use_cache": True,
    "batch_size": 1,
}


def prompt_contract_hash() -> str:
    contract = json.dumps(
        {"system_instruction": SYSTEM_INSTRUCTION, "user_template": USER_PROMPT_TEMPLATE},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()


def build_prompt(question: str, condition: str, documents: Sequence[str]) -> str:
    """Build the same user prompt shape for every retrieval condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

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
    return continuation.strip().splitlines()[0].strip() if continuation.strip() else ""


def looks_explanatory(continuation: str) -> bool:
    lines = [line.strip() for line in continuation.splitlines() if line.strip()]
    if len(lines) > 1:
        return True
    first = lines[0].lower() if lines else ""
    return bool(re.match(r"^(the answer|the .*\b(is|was|were|are)\b|because|according to|it is|this is)", first))


def generate_from_inputs(
    tokenizer: Any,
    model: Any,
    inputs: Any,
    *,
    max_new_tokens: int = GENERATION_SETTINGS["max_new_tokens"],
    max_input_tokens: int | None = None,
) -> str:
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids[0])
    batch_size = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else len(input_ids)
    if batch_size != GENERATION_SETTINGS["batch_size"]:
        raise ValueError(f"expected batch size 1, got {batch_size}")
    if max_input_tokens is not None and prompt_length + max_new_tokens > max_input_tokens:
        raise ValueError(
            "prompt plus generation exceeds the model context limit; "
            "refuse to truncate the question or answer instruction"
        )
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=GENERATION_SETTINGS["do_sample"],
            num_beams=GENERATION_SETTINGS["num_beams"],
            use_cache=GENERATION_SETTINGS["use_cache"],
        )
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def generate_continuation(
    tokenizer: Any,
    model: Any,
    user_prompt: str,
    *,
    max_new_tokens: int = GENERATION_SETTINGS["max_new_tokens"],
    do_sample: bool = GENERATION_SETTINGS["do_sample"],
    max_input_tokens: int | None = None,
) -> str:
    """Generate the decoded continuation after the input prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)

    return generate_from_inputs(
        tokenizer,
        model,
        inputs,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
    )


def generate_answer(
    tokenizer: Any,
    model: Any,
    user_prompt: str,
    *,
    max_new_tokens: int = GENERATION_SETTINGS["max_new_tokens"],
    do_sample: bool = GENERATION_SETTINGS["do_sample"],
    max_input_tokens: int | None = None,
) -> str:
    return parse_answer(
        generate_continuation(
            tokenizer,
            model,
            user_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            max_input_tokens=max_input_tokens,
        )
    )
