"""Align the question-acceptance ledger to the current frozen sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .fetch_wikipedia_corpus import load_jsonl, write_jsonl
except ImportError:
    from fetch_wikipedia_corpus import load_jsonl, write_jsonl


CHECK_FIELDS = (
    "grammar_meaningful", "answers_match_relation", "subject_title_correct",
    "unambiguous", "wikipedia_supports_fact",
)


def reconcile_review_rows(questions, reviews):
    existing = {row["question_id"]: row for row in reviews}
    result = []
    for question in questions:
        prior = existing.get(question["question_id"])
        row = {**(prior or {}), **question}
        if prior is None or not prior.get("reviewer_status"):
            for field in CHECK_FIELDS:
                row[field] = "pending"
            row["reviewer_status"] = "pending_ai_review"
            row["reviewer_notes"] = "Question entered the frozen sample after the original review; awaiting AI adjudication."
        result.append(row)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    review_path = root / "data/manifests/popqa_manual_review.jsonl"
    questions = load_jsonl(sample_path)
    reviews = load_jsonl(review_path)
    result = reconcile_review_rows(questions, reviews)
    write_jsonl(review_path, result)
    manifest_path = root / "data/manifests/popqa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest["manual_review_count"] = len(result)
    manifest["manual_review_status"] = "pending_ai_review" if any(row["reviewer_status"] != "accepted" for row in result) else "complete"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"review_rows": len(result), "pending": sum(row["reviewer_status"] != "accepted" for row in result)}))
