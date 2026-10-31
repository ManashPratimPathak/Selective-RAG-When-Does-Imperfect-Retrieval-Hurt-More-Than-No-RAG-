"""Replace adjudicated PopQA rows with evidence-backed, stratum-matched rows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .fetch_wikipedia_corpus import load_jsonl, write_jsonl
    from .finalize_phase3_questions import clarify_question, sanitize_aliases
except ImportError:
    from fetch_wikipedia_corpus import load_jsonl, write_jsonl
    from finalize_phase3_questions import clarify_question, sanitize_aliases


REPLACEMENTS = {
    1126311: 1409534, 2855671: 5760400, 5910134: 129760,
    1519349: 1488536, 4374873: 3111536, 5291073: 4939964,
    5840369: 401885, 5907056: 298575, 6028537: 906463,
    2985037: 84077, 5924855: 2442030, 4798831: 1887458,
    6435951: 2716696, 411235: 985634, 2734115: 156556,
    5222045: 3521531, 201734: 6288361, 316231: 1664593,
    4105002: 1163970, 130950: 3093779,
    129760: 6163158, 2491290: 4173872, 2951294: 2122202,
    3775555: 6261308,
}


def apply_replacements(questions, gold_pages, candidates, candidate_pages, replacements):
    old_by_id = {row["question_id"]: row for row in questions}
    used_ids = set(old_by_id)
    if len(set(replacements.values())) != len(replacements):
        raise ValueError("a candidate replacement cannot fill two slots")
    for old_id, new_id in replacements.items():
        if old_id not in old_by_id or new_id in used_ids or new_id not in candidates:
            raise ValueError(f"invalid replacement identity: {old_id} -> {new_id}")
        old = old_by_id[old_id]
        new = candidates[new_id]
        if old["relation"] != new["relation"] or old["popularity_group"] != new["popularity_group"]:
            raise ValueError(f"replacement changes relation or popularity group: {old_id} -> {new_id}")
        page = candidate_pages.get(new["wikipedia_title"])
        if not page or not page.get("revision_id") or not page.get("page_id"):
            raise ValueError(f"replacement lacks a revisioned page: {new_id}")
    updated_questions = []
    for old in questions:
        new_id = replacements.get(old["question_id"])
        if new_id is None:
            updated_questions.append(dict(old))
            continue
        new = candidates[new_id]
        updated = {
            key: old["split"] if key == "split" else old["sampling_seed"] if key == "sampling_seed" else new[key]
            for key in old if key != "source_question"
        }
        if {"question", "subject", "wikipedia_title"} <= updated.keys():
            updated = clarify_question(updated)
        if "accepted_answers" in updated:
            updated["accepted_answers"] = sanitize_aliases(updated["accepted_answers"])
        updated_questions.append(updated)
    active_titles = {row["wikipedia_title"] for row in updated_questions}
    updated_pages = [page for page in gold_pages if page["requested_title"] in active_titles]
    page_titles = {page["requested_title"] for page in updated_pages}
    for new_id in replacements.values():
        page = candidate_pages[candidates[new_id]["wikipedia_title"]]
        if page["requested_title"] not in page_titles:
            updated_pages.append(page)
            page_titles.add(page["requested_title"])
    page_ids = [str(page["page_id"]) for page in updated_pages]
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("replacement would duplicate a gold page ID")
    return updated_questions, updated_pages


def rebalance_distractors(distractors, gold_pages, candidates, candidate_pages, groups, *, preferred_ids=None):
    """Promote conflicting distractors to gold and backfill from unused PopQA pages."""
    gold_ids = {str(page["page_id"]) for page in gold_pages}
    promoted_ids = {str(page["page_id"]) for page in distractors} & gold_ids
    if not promoted_ids:
        return list(distractors), []
    occupied = {str(page["page_id"]) for page in distractors} | gold_ids
    updated = [page for page in distractors if str(page["page_id"]) not in promoted_ids]
    substitutions = []
    choices = list(preferred_ids) if preferred_ids is not None else sorted(candidates)
    for promoted_id in sorted(promoted_ids):
        group = groups[promoted_id]
        selected = None
        for candidate_id in choices:
            candidate = candidates.get(candidate_id)
            if candidate is None or candidate["popularity_group"] != group:
                continue
            page = candidate_pages.get(candidate["wikipedia_title"])
            if page is None or not page.get("revision_id") or str(page.get("page_id")) in occupied:
                continue
            selected = (candidate_id, page)
            break
        if selected is None:
            raise ValueError(f"no unused revisioned {group} distractor page for promoted page {promoted_id}")
        candidate_id, page = selected
        updated.append(page)
        occupied.add(str(page["page_id"]))
        substitutions.append({
            "promoted_page_id": promoted_id,
            "replacement_page_id": str(page["page_id"]),
            "replacement_title": page["requested_title"],
            "entity_group": group,
            "source_question_id": candidate_id,
        })
    return updated, substitutions


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    gold_path = root / "data/corpus/wikipedia_pages.jsonl"
    candidates = {row["question_id"]: row for row in load_jsonl(root / "data/frozen/popqa_replacement_candidates.jsonl")}
    candidate_pages = {page["requested_title"]: page for page in load_jsonl(root / "data/corpus/replacement_candidate_pages.jsonl")}
    questions = load_jsonl(sample_path)
    gold_pages = load_jsonl(gold_path)
    active_ids = {row["question_id"] for row in questions}
    replacements = {old: new for old, new in REPLACEMENTS.items() if old in active_ids}
    if not replacements:
        print("No unapplied evidence replacements")
        raise SystemExit(0)
    updated_questions, updated_pages = apply_replacements(questions, gold_pages, candidates, candidate_pages, replacements)
    distractor_path = root / "data/corpus/distractor_pages.jsonl"
    distractors = load_jsonl(distractor_path)
    distractor_manifest_path = root / "data/manifests/distractor_manifest.json"
    distractor_manifest = json.loads(distractor_manifest_path.read_text(encoding="utf-8"))
    groups = {str(row["final_page_id"]): row["entity_group"] for row in distractor_manifest["pages"] if row.get("status") == "fetched"}
    updated_distractors, substitutions = rebalance_distractors(
        distractors, updated_pages, candidates, candidate_pages, groups,
        preferred_ids=[5059618, 5924411, 4783910, 5343981, 2835997],
    )
    if len(updated_distractors) != len(distractors):
        raise ValueError("distractor replacement changed corpus size")
    old_by_id = {row["question_id"]: row for row in questions}
    new_by_id = {row["question_id"]: row for row in updated_questions}
    review_path = root / "data/manifests/popqa_manual_review.jsonl"
    review_by_id = {row["question_id"]: row for row in load_jsonl(review_path)}
    review_rows = []
    for row in updated_questions:
        old_id = next((old for old, new in replacements.items() if new == row["question_id"]), row["question_id"])
        review = dict(review_by_id.get(old_id, {}))
        review.update(row)
        if old_id in replacements:
            for field in ("grammar_meaningful", "answers_match_relation", "subject_title_correct", "unambiguous", "wikipedia_supports_fact"):
                review[field] = "pending"
            review["reviewer_status"] = "pending_ai_review"
            review["reviewer_notes"] = f"Replacement for {old_id}; awaiting passage-level AI adjudication."
        review_rows.append(review)
    replacement_log = [
        {
            "question_id": old_id,
            "replacement_id": new_id,
            "stage": "evidence_adjudication",
            "exclusion_reason": "incorrect_or_unsupported_subject_relation_answer",
            "popularity_group": old_by_id[old_id]["popularity_group"],
            "relation": old_by_id[old_id]["relation"],
            "split": old_by_id[old_id]["split"],
            "page_id": candidate_pages[candidates[new_id]["wikipedia_title"]]["page_id"],
            "revision_id": candidate_pages[candidates[new_id]["wikipedia_title"]]["revision_id"],
        }
        for old_id, new_id in replacements.items()
    ]
    write_jsonl(sample_path, updated_questions)
    write_jsonl(gold_path, updated_pages)
    if substitutions:
        write_jsonl(distractor_path, updated_distractors)
        promoted = {row["promoted_page_id"] for row in substitutions}
        distractor_manifest["pages"] = [
            row for row in distractor_manifest["pages"] if str(row.get("final_page_id")) not in promoted
        ]
        distractor_manifest["pages"].extend({
            "page_title": row["replacement_title"],
            "entity_group": row["entity_group"],
            "source_question_id": row["source_question_id"],
            "status": "fetched",
            "final_page_id": row["replacement_page_id"],
            "final_title": row["replacement_title"],
            "replacement_for_promoted_page_id": row["promoted_page_id"],
        } for row in substitutions)
        distractor_manifest["sha256"] = hashlib.sha256(distractor_path.read_bytes()).hexdigest()
        distractor_manifest["corpus_frozen_at"] = datetime.now(timezone.utc).isoformat()
        distractor_manifest["promoted_to_gold_replacements"] = distractor_manifest.get("promoted_to_gold_replacements", []) + substitutions
        distractor_manifest["overlap_with_answer_corpus"] = False
        distractor_manifest_path.write_text(json.dumps(distractor_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(review_path, review_rows)
    exclusion_path = root / "data/manifests/popqa_exclusion_log.jsonl"
    write_jsonl(exclusion_path, load_jsonl(exclusion_path) + replacement_log)
    manifest_path = root / "data/manifests/popqa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest["evidence_replacements"] = manifest.get("evidence_replacements", []) + replacement_log
    manifest["evidence_replacements_at"] = datetime.now(timezone.utc).isoformat()
    manifest["manual_review_status"] = "pending_ai_review"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"replacements": len(replacements), "questions": len(updated_questions), "gold_pages": len(updated_pages), "distractor_substitutions": substitutions}))
