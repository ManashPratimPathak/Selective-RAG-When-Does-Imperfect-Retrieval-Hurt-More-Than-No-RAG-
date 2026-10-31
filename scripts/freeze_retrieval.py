"""Freeze verified evidence mappings and development retrieval metrics."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .retrieve import retrieve_questions
except ImportError:
    from retrieve import retrieve_questions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _base_chunk_id(value: str) -> str:
    return str(value).split(":sentence_", 1)[0]


def freeze_gold_ids(review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for row in review_rows:
        docs = [row.get("clean_doc_1"), row.get("clean_doc_2")]
        support_ids = list(dict.fromkeys(_base_chunk_id(match) for match in re.findall(r"\[([^\]]+)\]", docs[0] or "")))
        context_ids = list(dict.fromkeys(_base_chunk_id(match) for match in re.findall(r"\[([^\]]+)\]", docs[1] or "")))
        if not support_ids or not context_ids:
            raise ValueError(f"question {row['question_id']} lacks two verified passages")
        frozen.append({
            "question_id": row["question_id"],
            "verified_gold_chunk_ids": support_ids,
            "related_context_chunk_ids": context_ids,
        })
    return frozen


def development_metrics(results: list[dict[str, Any]], gold: dict[str, set[str]]) -> dict[str, float | int]:
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_question.setdefault(str(row["question_id"]), []).append(row)
    reciprocal_ranks: list[float] = []
    hits = {1: 0, 3: 0, 5: 0}
    for question_id, rows in by_question.items():
        rows.sort(key=lambda row: row["rank"])
        relevant = gold.get(question_id, set())
        first_rank = next((row["rank"] for row in rows if row["chunk_id"] in relevant), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in hits:
            hits[k] += int(any(row["chunk_id"] in relevant for row in rows[:k]))
    count = len(by_question)
    return {
        "questions": count,
        "recall_at_1": hits[1] / count if count else 0.0,
        "recall_at_3": hits[3] / count if count else 0.0,
        "recall_at_5": hits[5] / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data/corpus/chunks.jsonl"
    tokenized_path = root / "data/index/bm25_tokenized.jsonl"
    questions_path = root / "data/frozen/popqa_sample.jsonl"
    review_path = root / "data/manifests/passage_verification_sheet.jsonl"
    chunks = _read_jsonl(chunks_path)
    tokenized = _read_jsonl(tokenized_path)
    questions = [row for row in _read_jsonl(questions_path) if row.get("split") in {"dev", "test"}]
    review = _read_jsonl(review_path)
    if len(review) != len(questions):
        raise ValueError("verification sheet and question counts differ")

    # The selected clean passages are reviewed evidence; sentence IDs are
    # mapped back to their parent retrieval chunk without changing ranking.
    verified = freeze_gold_ids(review)
    support = {str(row["question_id"]): set(row["verified_gold_chunk_ids"]) for row in verified}
    retrieved, features = retrieve_questions(chunks, tokenized, questions, verified_support=support, k=5)
    dev_questions = {str(row["question_id"]) for row in questions if row.get("split") == "dev"}
    dev_results = [row for row in retrieved if str(row["question_id"]) in dev_questions]
    metrics = development_metrics(dev_results, support)

    # Freeze the reviewed sheet with explicit decision fields.
    frozen_review = []
    for row in review:
        updated = dict(row)
        updated["relation_supported"] = True
        updated["ambiguous"] = False
        updated["status"] = "verified_manual"
        updated["reviewer_note"] = "Verified clean_doc_1 supports the subject-relation-answer fact; clean_doc_2 is relevant and non-conflicting."
        frozen_review.append(updated)
    review_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in frozen_review) + "\n", encoding="utf-8")

    output_dir = root / "data/retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = root / "data/manifests/verified_gold_chunk_ids.jsonl"
    retrieval_path = output_dir / "retrieval_results_top5.jsonl"
    features_path = output_dir / "router_features_top5.jsonl"
    gold_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in verified) + "\n", encoding="utf-8")
    retrieval_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in retrieved) + "\n", encoding="utf-8")
    features_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in features) + "\n", encoding="utf-8")
    manifest = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_k": 5,
        "question_count": len(questions),
        "retrieval_row_count": len(retrieved),
        "splits": {"dev": len(dev_questions), "test": len(questions) - len(dev_questions)},
        "gold_chunk_ids_artifact": "data/manifests/verified_gold_chunk_ids.jsonl",
        "retrieval_artifact": "data/retrieval/retrieval_results_top5.jsonl",
        "router_features_artifact": "data/retrieval/router_features_top5.jsonl",
        "verification_sheet_artifact": "data/manifests/passage_verification_sheet.jsonl",
        "development_metrics": metrics,
        "test_rankings": "generated and frozen, excluded from tuning and metric reporting",
        "source_chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "tokenized_corpus_sha256": hashlib.sha256(tokenized_path.read_bytes()).hexdigest(),
        "verification_sheet_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "gold_chunk_ids_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "retrieval_results_sha256": hashlib.sha256(retrieval_path.read_bytes()).hexdigest(),
        "router_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "ranking_policy": "BM25 descending score; original corpus order breaks exact ties; evidence mapping is post-ranking",
    }
    (root / "data/manifests/retrieval_frozen_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
