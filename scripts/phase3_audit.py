"""Check the frozen Phase 3 corpus and manual passage-review contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _document_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("[") or "]" not in value:
        return None
    return value[1 : value.index("]")]


def _overlapping_passages(first: str, second: str) -> bool:
    first_base, _, first_sentence = first.partition(":sentence_")
    second_base, _, second_sentence = second.partition(":sentence_")
    return first_base == second_base and (not first_sentence or not second_sentence or first_sentence == second_sentence)


def _usable_gold_text(text: str) -> bool:
    folded = " ".join(str(text).casefold().split())
    return len(folded) >= 100 and "this page is a redirect" not in folded and "this disambiguation page" not in folded


def _passage_matches_chunk(document: str, chunk_lookup: dict[str, dict[str, Any]]) -> bool:
    document_id = _document_id(document)
    if not document_id:
        return False
    base_id, separator, sentence_index = document_id.partition(":sentence_")
    chunk = chunk_lookup.get(base_id)
    if chunk is None:
        return False
    passage_text = document.partition("] ")[2].strip()
    if separator:
        if not sentence_index.isdigit():
            return False
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))]
        index = int(sentence_index)
        return index < len(sentences) and passage_text == sentences[index]
    return passage_text == str(chunk.get("text", "")).strip()


def audit_phase3(
    questions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    distractors: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    review: list[dict[str, Any]],
    question_review: list[dict[str, Any]],
    *,
    expected_distractors: int = 500,
) -> dict[str, Any]:
    question_ids = [row["question_id"] for row in questions]
    raw_pages = gold + distractors
    raw_ids = [str(page.get("page_id")) for page in raw_pages]
    raw_titles = {page.get("requested_title") for page in gold}
    cleaned_ids = {str(page.get("page_id")) for page in cleaned}
    chunk_ids = {str(page.get("page_id")) for page in chunks}
    chunk_lookup = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    gold_id_by_title = {page.get("requested_title"): str(page.get("page_id")) for page in gold}
    gold_ids = set(gold_id_by_title.values())
    unusable_gold_page_ids = [str(page.get("page_id")) for page in cleaned if str(page.get("page_id")) in gold_ids and not _usable_gold_text(page.get("text", ""))]
    question_by_id = {row["question_id"]: row for row in questions}
    review_by_id = {row.get("question_id"): row for row in review}
    question_review_by_id = {row.get("question_id"): row for row in question_review}
    accepted_question_ids = {
        question_id for question_id, row in question_review_by_id.items()
        if row.get("reviewer_status") == "accepted"
    }
    status_counts = Counter(row.get("status", "missing_status") for row in review)
    manual_verified = 0
    invalid_passage_ids: list[Any] = []
    for question_id in question_ids:
        row = review_by_id.get(question_id)
        if row is None or row.get("status") != "verified_manual":
            continue
        first = _document_id(row.get("clean_doc_1"))
        second = _document_id(row.get("clean_doc_2"))
        if (
            not first or not second or first == second or _overlapping_passages(first, second)
            or str(chunk_lookup.get(first.split(":sentence_")[0], {}).get("page_id")) != gold_id_by_title.get(question_by_id[question_id]["wikipedia_title"])
            or str(chunk_lookup.get(second.split(":sentence_")[0], {}).get("page_id")) != gold_id_by_title.get(question_by_id[question_id]["wikipedia_title"])
            or not _passage_matches_chunk(row["clean_doc_1"], chunk_lookup)
            or not _passage_matches_chunk(row["clean_doc_2"], chunk_lookup)
            or row.get("relation_supported") is not True
            or row.get("ambiguous") is not False
            or not str(row.get("reviewer_note") or "").strip()
        ):
            invalid_passage_ids.append(question_id)
            continue
        manual_verified += 1

    checks = {
        "question_count_240": len(questions) == 240 and len(set(question_ids)) == 240,
        "accepted_question_review_coverage": len(question_review) == 240
        and set(question_review_by_id) == set(question_ids)
        and len(accepted_question_ids) == 240
        and all(question_review_by_id[question_id].get("question") == question_by_id[question_id].get("question") for question_id in question_ids),
        "gold_page_coverage": all(row["wikipedia_title"] in raw_titles for row in questions),
        "distractor_count": len(distractors) == expected_distractors,
        "revision_tracked_unique_pages": len(set(raw_ids)) == len(raw_ids)
        and all(page.get("page_id") and page.get("revision_id") and page.get("source_url") for page in raw_pages),
        "cleaned_page_coverage": len(cleaned) == len(raw_pages) and cleaned_ids == set(raw_ids),
        "gold_articles_substantive": not unusable_gold_page_ids,
        "chunk_page_coverage": chunk_ids == set(raw_ids),
        "review_row_coverage": len(review) == 240 and set(review_by_id) == set(question_ids),
        "passage_question_alignment": all(
            review_by_id.get(question_id, {}).get("question") == question_by_id[question_id].get("question")
            for question_id in question_ids
        ),
        "two_manually_verified_passages_each": manual_verified == 240 and not invalid_passage_ids,
    }
    return {
        "status": "complete" if all(checks.values()) else "incomplete",
        "checks": checks,
        "question_count": len(questions),
        "gold_page_count": len(gold),
        "distractor_page_count": len(distractors),
        "cleaned_page_count": len(cleaned),
        "chunk_count": len(chunks),
        "manual_verified_questions": manual_verified,
        "accepted_question_review_count": len(accepted_question_ids & set(question_ids)),
        "missing_question_review_ids": sorted(set(question_ids) - set(question_review_by_id)),
        "invalid_manual_review_question_ids": invalid_passage_ids,
        "unusable_gold_page_ids": unusable_gold_page_ids,
        "review_status_counts": dict(sorted(status_counts.items())),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = {
        "questions": "data/frozen/popqa_sample.jsonl",
        "gold": "data/corpus/wikipedia_pages.jsonl",
        "distractors": "data/corpus/distractor_pages.jsonl",
        "cleaned": "data/corpus/cleaned_pages.jsonl",
        "chunks": "data/corpus/chunks.jsonl",
        "review": "data/manifests/passage_verification_sheet.jsonl",
        "question_review": "data/manifests/popqa_manual_review.jsonl",
    }
    rows = {name: _read_jsonl(root / relative) for name, relative in artifacts.items()}
    audit = audit_phase3(**rows)
    audit["created_at"] = datetime.now(timezone.utc).isoformat()
    audit["artifacts"] = {
        name: {"path": relative, "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest()}
        for name, relative in artifacts.items()
    }
    output = root / "data/manifests/phase3_corpus_manifest.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
