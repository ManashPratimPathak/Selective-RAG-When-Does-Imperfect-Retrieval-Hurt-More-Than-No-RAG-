"""Create or verify the immutable Step 5.4 prompt manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from generate import GENERATION_SETTINGS, SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE, prompt_contract_hash


def manifest(root: Path) -> dict:
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    return {
        "freeze_version": 1,
        "system_instruction": SYSTEM_INSTRUCTION,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "prompt_hash_algorithm": "sha256(canonical_json(system_instruction,user_prompt_template))",
        "prompt_hash": prompt_contract_hash(),
        "decoding": {
            **GENERATION_SETTINGS,
            "max_input_tokens": int(config["max_input_tokens"]),
        },
        "quantization": config["quantization"],
        "model": {
            "id": config["model"],
            "revision": str(config["model_revision"]),
            "tokenizer_revision": str(config["tokenizer_revision"]),
        },
        "document_order": {
            "clean_rag": "verified clean_doc_1, then clean_doc_2",
            "imperfect_rag": "frozen BM25 rank 1, then rank 2, then rank 3",
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the existing manifest without rewriting it")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "data/manifests/prompt_freeze.json"
    current = manifest(root)
    if args.check:
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key in current:
            if key == "frozen_at":
                continue
            if saved.get(key) != current[key]:
                raise SystemExit(f"prompt freeze mismatch: {key}")
        print(f"Prompt freeze verified: {path}")
        return
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prompt frozen: {path}")


if __name__ == "__main__":
    main()
