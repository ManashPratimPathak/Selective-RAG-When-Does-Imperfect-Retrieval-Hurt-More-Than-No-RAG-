"""Correct unusable Wikipedia shell titles without changing question identity."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from .fetch_wikipedia_corpus import fetch_page, load_jsonl, write_jsonl
except ImportError:
    from fetch_wikipedia_corpus import fetch_page, load_jsonl, write_jsonl


CORRECTIONS = {
    "Laika (company)": "Laika, LLC",
    "The Signal (2007 film)": "The Signal (2007 American film)",
    "Basic (film)": "Basic (2003 film)",
    "The Kingdom (film)": "The Kingdom (2007 film)",
    "That Certain Thing": "That Certain Thing (film)",
    "Our Land (film)": "Our Land (2006 film)",
}


def apply_title_corrections(questions, pages, fetched_pages):
    old_pages = {page["requested_title"]: page for page in pages}
    updated_questions = []
    corrections = []
    for question in questions:
        old_title = question["wikipedia_title"]
        if old_title not in fetched_pages:
            updated_questions.append(dict(question))
            continue
        if old_title not in old_pages:
            raise ValueError(f"missing old gold page for {old_title}")
        replacement = fetched_pages[old_title]
        new_title = replacement["requested_title"]
        updated_questions.append({**question, "wikipedia_title": new_title})
        corrections.append({
            "question_id": question["question_id"],
            "old_title": old_title,
            "new_title": new_title,
            "old_page_id": old_pages[old_title]["page_id"],
            "new_page_id": replacement["page_id"],
        })
    updated_pages = [fetched_pages.get(page["requested_title"], page) for page in pages]
    if len({str(page["page_id"]) for page in updated_pages}) != len(updated_pages):
        raise ValueError("title correction would duplicate a gold page ID")
    return updated_questions, updated_pages, corrections


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    corpus_path = root / "data/corpus/wikipedia_pages.jsonl"
    questions = load_jsonl(sample_path)
    pages = load_jsonl(corpus_path)
    active_titles = {question["wikipedia_title"] for question in questions}
    fetched = {}
    with requests.Session() as session:
        for old_title, new_title in CORRECTIONS.items():
            if old_title not in active_titles:
                continue
            page, outcome = fetch_page(session, new_title)
            if page is None or page.get("requested_title") != new_title:
                raise RuntimeError(f"could not correct {old_title}: {outcome}")
            fetched[old_title] = page
    updated_questions, updated_pages, corrections = apply_title_corrections(questions, pages, fetched)
    write_jsonl(sample_path, updated_questions)
    write_jsonl(corpus_path, updated_pages)
    review_path = root / "data/manifests/popqa_manual_review.jsonl"
    review = load_jsonl(review_path)
    by_id = {question["question_id"]: question for question in updated_questions}
    corrected_ids = {entry["question_id"] for entry in corrections}
    for row in review:
        if row["question_id"] in corrected_ids:
            row["wikipedia_title"] = by_id[row["question_id"]]["wikipedia_title"]
            row["wikipedia_supports_fact"] = "pending"
            row["reviewer_status"] = "pending_ai_review"
            row["reviewer_notes"] = "Corrected a shell/disambiguation title; recheck evidence before acceptance."
    write_jsonl(review_path, review)
    manifest_path = root / "data/manifests/popqa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest["gold_title_corrections"] = corrections
    manifest["gold_title_corrections_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"corrected_titles": len(corrections), "gold_page_count": len(updated_pages)}))
