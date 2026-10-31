"""Prepare a human verification sheet for candidate gold passages."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .gold_candidates import normalize_alias
except ImportError:
    from gold_candidates import normalize_alias


def _has_alias(candidate: dict[str, Any], answers: list[Any]) -> bool:
    text = f" {normalize_alias(candidate.get('text', ''))} "
    return any(f" {normalize_alias(answer)} " in text for answer in answers if normalize_alias(answer))


def supplemental_candidate(chunk: dict[str, Any], question: dict[str, Any], chunk_id: str, text: str, rank: int) -> dict[str, Any]:
    return {
        **chunk,
        "chunk_id": chunk_id,
        "text": text,
        "candidate_rank": rank,
        "matched_aliases": question.get("accepted_answers", []),
        "relation_keyword": question.get("relation", ""),
    }

def build_verification_row(question: dict[str, Any], candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build one review row; lexical candidates never become verified facts."""
    ordered: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.get("candidate_rank", 10**9)):
        chunk_id = str(candidate["chunk_id"])
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        ordered.append(candidate)

    selected = ordered[:1]
    if selected:
        first_base = str(selected[0]["chunk_id"]).partition(":sentence_")[0]
        separate = next(
            (candidate for candidate in ordered[1:] if str(candidate["chunk_id"]).partition(":sentence_")[0] != first_base),
            None,
        )
        if separate is not None:
            selected.append(separate)
        else:
            sentences = [candidate for candidate in ordered if str(candidate["chunk_id"]).startswith(first_base + ":sentence_")]
            sentences.sort(key=lambda candidate: not _relation_supported_candidate(candidate))
            selected = sentences[:2] if len(sentences) >= 2 else selected

    docs = [f"[{candidate['chunk_id']}] {candidate['text']}" for candidate in selected]
    while len(docs) < 2:
        docs.append(None)
    alias_present = {
        "clean_doc_1": bool(selected and _has_alias(selected[0], question.get("accepted_answers", []))),
        "clean_doc_2": bool(len(selected) > 1 and _has_alias(selected[1], question.get("accepted_answers", []))),
    }
    if not selected:
        status = "rejected_no_candidate_passages"
        note = "No lexical candidate was found; do not infer support."
    elif len(selected) < 2:
        status = "needs_second_passage"
        note = "Fewer than two distinct lexical candidates were found; locate a second relevant, non-conflicting passage."
    else:
        status = "needs_manual_review"
        note = "Manually confirm clean_doc_1 establishes subject–relation–answer; confirm clean_doc_2 is relevant and non-conflicting. Reject answer-only matches."
    return {
        "question_id": question["question_id"],
        "question": question.get("question"),
        "accepted_answers": question.get("accepted_answers", []),
        "clean_doc_1": docs[0],
        "clean_doc_2": docs[1],
        "alias_present": alias_present,
        "relation_supported": None,
        "ambiguous": None,
        "reviewer_note": note,
        "status": status,
    }


def finalize_verification_row(row: dict[str, Any], *, relation_supported: bool) -> dict[str, Any]:
    """Record a lexical screen without making a review decision."""
    final = dict(row)
    final["automated_relation_match"] = bool(relation_supported)
    if final["status"] == "needs_manual_review":
        final["reviewer_note"] = (
            "Lexical relation match found; manually verify both passages and the question."
            if relation_supported else
            "No lexical relation match found; inspect the source before excluding this question."
        )
    return final


def carry_forward_review(generated: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    fields = ("question_id", "question", "accepted_answers", "clean_doc_1", "clean_doc_2")
    if previous and previous.get("status") in {"verified_manual", "rejected_manual"} and all(
        generated.get(field) == previous.get(field) for field in fields
    ):
        return previous
    return generated


def promote_candidate(candidates: list[dict[str, Any]], chunk_id: str) -> list[dict[str, Any]]:
    if not any(str(candidate["chunk_id"]) == chunk_id for candidate in candidates):
        raise ValueError(f"review override chunk is missing: {chunk_id}")
    return sorted(candidates, key=lambda candidate: str(candidate["chunk_id"]) != chunk_id)


def _relation_supported_candidate(candidate: dict[str, Any]) -> bool:
    relation = normalize_alias(candidate.get("relation_keyword", ""))
    aliases = [normalize_alias(alias) for alias in candidate.get("matched_aliases", [])]
    if not relation or not aliases:
        return False

    relation_hints = {
        "director": ("director", "directed"),
        "producer": ("producer", "produced"),
        "screenwriter": ("screenwriter", "screenplay by", "written by", "writer"),
        "composer": ("composer", "composed", "music by", "score by"),
        "author": ("author", "written by", "novel by", "story by"),
        "father": ("father", "son of", "daughter of"),
        "capital": ("capital",),
        "country": ("country", "in the "),
        "place of birth": ("born", "birth"),
        "religion": ("religion", "anglican", "catholic", "christian"),
        "sport": ("sport", "football", "soccer", "footballer", "athlete", "bobsledder", "rugby"),
        "genre": ("genre", "film", "novel", "album", "music", "comedy", "rock", "painting", "game"),
        "occupation": ("journalist", "pianist", "physician", "astronaut", "manager", "actress", "actor", "diplomat", "umpire", "composer", "bishop", "writer", "singer", "businessman"),
    }
    hints = relation_hints.get(relation, (relation,))

    def alias_present(normalized_sentence: str, alias: str) -> bool:
        alias_tokens = alias.split()
        sentence_tokens = normalized_sentence.split()
        for start, token in enumerate(sentence_tokens):
            if token != alias_tokens[0]:
                continue
            cursor = start
            for expected in alias_tokens[1:]:
                matches = [
                    index
                    for index in range(cursor + 1, min(len(sentence_tokens), cursor + 4))
                    if sentence_tokens[index] == expected
                ]
                if not matches:
                    break
                cursor = matches[0]
            else:
                return True
        return False

    import re

    for sentence in re.split(r"(?<=[.!?])\s+", candidate.get("text", "")):
        normalized = normalize_alias(sentence)
        if any(alias_present(normalized, alias) for alias in aliases) and any(hint in normalized for hint in hints):
            return True
    return False


relation_supported_candidate = _relation_supported_candidate


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    questions = _load_jsonl(root / "data/frozen/popqa_sample.jsonl")
    candidate_rows = _load_jsonl(root / "data/manifests/gold_passage_candidates.jsonl")
    chunks = _load_jsonl(root / "data/corpus/chunks.jsonl")
    overrides_path = root / "data/manifests/passage_candidate_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    output = root / "data/manifests/passage_verification_sheet.jsonl"
    previous_reviews = {row["question_id"]: row for row in _load_jsonl(output)} if output.exists() else {}
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        by_question[str(candidate["question_id"])].append(candidate)
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_title[normalize_alias(chunk.get("requested_title") or chunk.get("page_title", ""))].append(chunk)
    sheet = []
    for question in questions:
        candidates = by_question.get(str(question["question_id"]), [])
        supported = [candidate for candidate in candidates if _relation_supported_candidate(candidate)]
        unsupported = [candidate for candidate in candidates if candidate not in supported]
        selected_ids = {str(candidate["chunk_id"]) for candidate in candidates}
        supplemental: list[dict[str, Any]] = []
        page_chunks = by_title.get(normalize_alias(question["wikipedia_title"]), [])
        for chunk in page_chunks:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id not in selected_ids:
                supplemental.append(supplemental_candidate(chunk, question, chunk_id, chunk["text"], 10**8 + int(chunk.get("chunk_index", 0))))
            # A single 100-word chunk can contain multiple distinct clean
            # passages. Expose sentence-level context for clean_doc_2 while
            # preserving the original chunk as the primary evidence unit.
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))]
            for sentence_index, sentence in enumerate(sentences):
                if len(sentence.split()) < 8:
                    continue
                supplemental.append(supplemental_candidate(
                    chunk, question, f"{chunk_id}:sentence_{sentence_index:02d}", sentence,
                    10**8 + int(chunk.get("chunk_index", 0)) * 100 + sentence_index,
                ))
        ordered = supported + unsupported + supplemental
        preferred_id = overrides.get(str(question["question_id"]))
        if preferred_id:
            ordered = promote_candidate(ordered, preferred_id)
        ordered = [{**candidate, "candidate_rank": rank} for rank, candidate in enumerate(ordered, start=1)]
        row = build_verification_row(question, ordered)
        first_id = row["clean_doc_1"].partition("]")[0][1:] if row["clean_doc_1"] else None
        first = next((candidate for candidate in ordered if str(candidate["chunk_id"]) == first_id), None)
        generated = finalize_verification_row(row, relation_supported=_relation_supported_candidate(first) if first else False)
        sheet.append(carry_forward_review(generated, previous_reviews.get(question["question_id"])))

    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in sheet) + "\n", encoding="utf-8")
    counts = defaultdict(int)
    for row in sheet:
        counts[row["status"]] += 1
    manifest = {
        "status": "automated_screening_pending_manual_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": "data/manifests/passage_verification_sheet.jsonl",
        "question_count": len(sheet),
        "status_counts": dict(sorted(counts.items())),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "protocol": "Candidate pairs and lexical relation matches are automated screening only; a reviewer must verify the fact and both passages before acceptance or exclusion.",
    }
    (root / "data/manifests/passage_verification_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
