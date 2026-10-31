"""Build the deterministic BM25 corpus artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from .tokenization import tokenize
except ImportError:  # Direct execution via python scripts/05_build_bm25.py
    from tokenization import tokenize


TOKENIZATION_SETTINGS = {
    "lowercase": True,
    "punctuation": "replace_with_spaces",
    "retained_character_categories": ["letters", "numbers"],
    "collapse_whitespace": True,
    "stemming": False,
    "stop_word_removal": False,
}


@dataclass(frozen=True)
class IndexBuildResult:
    corpus_size: int
    tokenized_corpus_path: str
    manifest_path: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(
    chunks_path: Path,
    tokenized_corpus_path: Path,
    manifest_path: Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> IndexBuildResult:
    chunks = _read_jsonl(chunks_path)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    tokenized_rows = [{"chunk_id": chunk_id, "tokens": tokenize(chunk.get("indexed_text", chunk.get("text", "")))} for chunk_id, chunk in zip(chunk_ids, chunks)]

    # Build once to validate the exact corpus supplied to the persisted metadata.
    BM25Okapi([row["tokens"] for row in tokenized_rows], k1=k1, b=b)

    tokenized_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized_corpus_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in tokenized_rows) + "\n",
        encoding="utf-8",
    )
    source_manifest = chunks_path.parent.parent / "manifests" / "chunk_corpus_manifest.json"
    corpus_manifest_sha256 = hashlib.sha256(
        source_manifest.read_bytes() if source_manifest.exists() else chunks_path.read_bytes()
    ).hexdigest()
    manifest = {
        "status": "frozen",
        "source_corpus": str(chunks_path),
        "tokenized_corpus": str(tokenized_corpus_path),
        "corpus_size": len(chunks),
        "corpus_order": list(range(len(chunks))),
        "chunk_ids": chunk_ids,
        "tokenization": TOKENIZATION_SETTINGS,
        "bm25_parameters": {"k1": k1, "b": b},
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "corpus_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "tokenized_corpus_sha256": hashlib.sha256(tokenized_corpus_path.read_bytes()).hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IndexBuildResult(len(chunks), str(tokenized_corpus_path), str(manifest_path))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    result = build_index(
        root / "data/corpus/chunks.jsonl",
        root / "data/index/bm25_tokenized.jsonl",
        root / "data/manifests/bm25_manifest.json",
        k1=float(config.get("bm25_k1", 1.5)),
        b=float(config.get("bm25_b", 0.75)),
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
