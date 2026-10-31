"""Make the selected PopQA questions unambiguous without losing source wording."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from .fetch_wikipedia_corpus import load_jsonl, write_jsonl
except ImportError:
    from fetch_wikipedia_corpus import load_jsonl, write_jsonl


def sanitize_aliases(aliases: list[str]) -> list[str]:
    """Rights identifiers are not answer aliases, even when PopQA lists them."""
    return list(dict.fromkeys(
        str(alias).strip() for alias in aliases
        if str(alias).strip() and not re.search(r"\d{5,}|\bIPI\b", str(alias), re.IGNORECASE)
    ))


def clarify_question(row: dict) -> dict:
    result = dict(row)
    subject = str(row["subject"])
    title = str(row["wikipedia_title"])
    original = str(row.get("source_question") or row["question"])
    if subject.casefold() != title.casefold():
        match = re.search(rf"(?<!\w){re.escape(subject)}(?!\w)", original, re.IGNORECASE)
        if match is None:
            raise ValueError(f"question {row.get('question_id')} does not mention subject {subject!r}")
        result["source_question"] = original
        result["question"] = original[:match.start()] + title + original[match.end():]
    refinements = {
        4145108: "Who is the adoptive father of Trace Cyrus?",
        930027: "Who directed the first series of The Missing (British TV series)?",
    }
    if row.get("question_id") in refinements:
        result["source_question"] = original
        result["question"] = refinements[row["question_id"]]
    return result


def reconcile_review_row(question: dict, previous: dict) -> dict:
    changed = any(previous.get(field) != question.get(field) for field in ("question", "accepted_answers", "wikipedia_title"))
    review = {**previous, **question}
    if changed:
        for field in ("grammar_meaningful", "answers_match_relation", "subject_title_correct", "unambiguous", "wikipedia_supports_fact"):
            review[field] = "pending"
        review["reviewer_status"] = "pending_ai_review"
        review["reviewer_notes"] = "Pending passage-level review of clarified frozen question."
    return review


def clarification_ids(rows: list[dict]) -> list[int]:
    return [row["question_id"] for row in rows if row.get("source_question") and row["source_question"] != row["question"]]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    ledger_path = root / "data/manifests/popqa_manual_review.jsonl"
    manifest_path = root / "data/manifests/popqa_manifest.json"
    sample = load_jsonl(sample_path)
    ledger = {row["question_id"]: row for row in load_jsonl(ledger_path)}
    filtered = []
    for original in sample:
        row = clarify_question(original)
        row["accepted_answers"] = sanitize_aliases(row["accepted_answers"])
        if row["accepted_answers"] != original["accepted_answers"]:
            filtered.append(row["question_id"])
        ledger[row["question_id"]] = reconcile_review_row(row, ledger[row["question_id"]])
    write_jsonl(sample_path, sample := [clarify_question({**row, "accepted_answers": sanitize_aliases(row["accepted_answers"])}) for row in sample])
    write_jsonl(ledger_path, [ledger[row["question_id"]] for row in sample])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clarified = clarification_ids(sample)
    filtered = sorted(set(manifest.get("alias_filters", {}).get("question_ids", [])) | set(filtered))
    manifest["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest["columns"] = list(sample[0])
    manifest["freeze_schema"] = list(sample[0])
    manifest["question_clarifications"] = {
        "count": len(clarified),
        "question_ids": clarified,
        "method": "Replace the source subject label with its pinned Wikipedia title; specify adoptive father or first series for two ambiguous relations; retain source_question for provenance.",
    }
    manifest["alias_filters"] = {"count": len(filtered), "question_ids": filtered, "rule": "Exclude external rights identifiers from answer aliases."}
    manifest["manual_review_status"] = "pending_ai_passage_review"
    manifest["manual_review_count"] = len(sample)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"clarified": len(clarified), "alias_filtered": len(filtered), "questions": len(sample)}))


if __name__ == "__main__":
    main()
