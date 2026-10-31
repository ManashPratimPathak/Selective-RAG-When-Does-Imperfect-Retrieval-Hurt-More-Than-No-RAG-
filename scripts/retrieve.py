"""Retrieve deterministic top-k BM25 passages and router features."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from .tokenization import tokenize
except ImportError:  # Direct execution via python scripts/06_retrieve.py
    from tokenization import tokenize


def retrieve_questions(
    chunks: list[dict[str, Any]],
    tokenized: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    verified_support: dict[str, set[str]] | None = None,
    k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(chunks) != len(tokenized) or [str(row["chunk_id"]) for row in chunks] != [str(row["chunk_id"]) for row in tokenized]:
        raise ValueError("chunk and tokenized corpus order do not match")
    if k < 1:
        raise ValueError("k must be positive")
    bm25 = BM25Okapi([row["tokens"] for row in tokenized])
    support = verified_support or {}
    retrieved: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for question in questions:
        question_tokens = tokenize(question["question"])
        scores = bm25.get_scores(question_tokens)
        ranked = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k]
        question_id = question["question_id"]
        support_ids = support.get(str(question_id), set())
        for rank, index in enumerate(ranked, start=1):
            chunk = chunks[index]
            retrieved.append({
                "question_id": question_id,
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "page_title": chunk["page_title"],
                "bm25_score": float(scores[index]),
                "chunk_text": chunk["text"],
                "is_verified_support": str(chunk["chunk_id"]) in support_ids,
            })
        top_tokens = tokenized[ranked[0]]["tokens"] if ranked else []
        question_vocab = set(question_tokens)
        overlap = len(question_vocab & set(top_tokens)) / len(question_vocab) if question_vocab else 0.0
        s1 = float(scores[ranked[0]]) if ranked else 0.0
        s2 = float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
        features.append({
            "question_id": question_id,
            "split": question.get("split"),
            "s1": s1,
            "s2": s2,
            "score_margin": s1 - s2,
            "query_top_passage_token_overlap": overlap,
            "question_token_count": len(question_tokens),
        })
    return retrieved, features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verified_support(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    support: dict[str, set[str]] = {}
    for row in rows:
        if row.get("status") not in {"accepted", "verified", "manually_verified"} and row.get("relation_supported") is not True:
            continue
        ids = re.findall(r"\[([^\]]+)\]", f"{row.get('clean_doc_1', '')} {row.get('clean_doc_2', '')}")
        support.setdefault(str(row["question_id"]), set()).update(ids)
    return support


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config: dict[str, Any] = {}
    config_path = root / "config.yaml"
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    chunks_path = root / "data/corpus/chunks.jsonl"
    tokenized_path = root / "data/index/bm25_tokenized.jsonl"
    questions_path = root / "data/frozen/popqa_sample.jsonl"
    review_path = root / "data/manifests/passage_verification_sheet.jsonl"
    chunks = _read_jsonl(chunks_path)
    tokenized = _read_jsonl(tokenized_path)
    questions = [row for row in _read_jsonl(questions_path) if row.get("split") in {"dev", "test"}]
    support = _verified_support(_read_jsonl(review_path)) if review_path.exists() else {}
    retrieved, features = retrieve_questions(
        chunks, tokenized, questions, verified_support=support, k=int(config.get("retrieval_k", 3))
    )
    output_dir = root / "data/retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = output_dir / "retrieval_results.jsonl"
    features_path = output_dir / "router_features.jsonl"
    retrieval_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in retrieved) + "\n", encoding="utf-8")
    features_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in features) + "\n", encoding="utf-8")
    manifest = {
        "status": "frozen",
        "retrieval_k": int(config.get("retrieval_k", 3)),
        "question_count": len(questions),
        "retrieval_row_count": len(retrieved),
        "splits": {split: sum(row.get("split") == split for row in questions) for split in ("dev", "test")},
        "retrieval_artifact": "data/retrieval/retrieval_results.jsonl",
        "router_features_artifact": "data/retrieval/router_features.jsonl",
        "retrieval_sha256": hashlib.sha256(retrieval_path.read_bytes()).hexdigest(),
        "router_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
    }
    (root / "data/manifests/retrieval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
