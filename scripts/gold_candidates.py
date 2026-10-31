"""Find and rank candidate gold passages without asserting relational support."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def normalize_alias(value: str) -> str:
    """Normalize an alias for case-, punctuation-, and whitespace-insensitive matching."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> list[str]:
    return normalize_alias(value).split()


def _occurrence_starts(tokens: list[str], phrase: list[str]) -> list[int]:
    if not phrase or len(phrase) > len(tokens):
        return []
    return [
        index
        for index in range(len(tokens) - len(phrase) + 1)
        if tokens[index : index + len(phrase)] == phrase
    ]


def _fuzzy_occurrence_starts(tokens: list[str], phrase: list[str], max_gap: int = 2) -> list[int]:
    """Find ordered phrase tokens with a small wording gap between them."""
    if not phrase:
        return []
    starts: list[int] = []
    for start, token in enumerate(tokens):
        if token != phrase[0]:
            continue
        cursor = start
        for expected in phrase[1:]:
            next_positions = [index for index in range(cursor + 1, min(len(tokens), cursor + max_gap + 2)) if tokens[index] == expected]
            if not next_positions:
                break
            cursor = next_positions[0]
        else:
            starts.append(start)
    return starts


def _distance(subject_tokens: list[str], answer_tokens: list[str], tokens: list[str]) -> int | None:
    subject_starts = _occurrence_starts(tokens, subject_tokens)
    answer_starts = _occurrence_starts(tokens, answer_tokens)
    if not subject_starts or not answer_starts:
        return None
    return min(abs(subject_start - answer_start) for subject_start in subject_starts for answer_start in answer_starts)


def _aliases(question: dict[str, Any]) -> list[str]:
    values = question.get("accepted_answers", [])
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if str(value).strip()]


def rank_question_candidates(question: dict[str, Any], chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return alias-containing chunks from the question's subject page, ranked."""
    page_title = normalize_alias(question["wikipedia_title"])
    aliases = _aliases(question)
    alias_phrases = []
    for alias in aliases:
        tokens = _tokens(alias)
        variants = [tokens]
        if len(tokens) >= 2:
            # Wikipedia prose commonly drops geographic or title suffixes
            # while retaining the identifying first two name tokens.
            variants.append(tokens[:2])
        alias_phrases.append((alias, variants))
    relation_phrase = _tokens(question.get("relation", ""))
    subject_phrase = _tokens(question.get("subject") or question["wikipedia_title"])
    candidates: list[dict[str, Any]] = []

    for chunk in chunks:
        if normalize_alias(chunk.get("requested_title") or chunk.get("page_title", "")) != page_title:
            continue
        tokens = _tokens(chunk.get("text", ""))
        matched_aliases: list[str] = []
        alias_occurrences = 0
        answer_distance: int | None = None
        for alias, variants in alias_phrases:
            variant_starts = [_fuzzy_occurrence_starts(tokens, phrase) for phrase in variants]
            starts = max(variant_starts, key=len)
            if starts:
                matched_aliases.append(alias)
                alias_occurrences += len(starts)
                distances = [
                    distance
                    for phrase, occurrences in zip(variants, variant_starts)
                    if occurrences
                    for distance in [_distance(subject_phrase, phrase, tokens)]
                    if distance is not None
                ]
                distance = min(distances) if distances else None
                if distance is not None and (answer_distance is None or distance < answer_distance):
                    answer_distance = distance
        if not matched_aliases:
            continue
        relation_occurrences = len(_occurrence_starts(tokens, relation_phrase))
        candidates.append(
            {
                **chunk,
                "question_id": question["question_id"],
                "accepted_aliases": aliases,
                "normalized_aliases": [normalize_alias(alias) for alias in aliases],
                "matched_aliases": matched_aliases,
                "alias_occurrences": alias_occurrences,
                "relation_keyword": normalize_alias(question.get("relation", "")),
                "relation_occurrences": relation_occurrences,
                "subject_answer_distance": answer_distance,
                # This is deliberately not inferred from lexical evidence.
                "relational_support": False,
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["alias_occurrences"],
            -candidate["relation_occurrences"],
            candidate["subject_answer_distance"] if candidate["subject_answer_distance"] is not None else 10**9,
            candidate["chunk_index"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["candidate_rank"] = rank
    return candidates


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    questions = _load_jsonl(root / "data/frozen/popqa_sample.jsonl")
    chunks = _load_jsonl(root / "data/corpus/chunks.jsonl")
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_title[normalize_alias(chunk.get("requested_title") or chunk["page_title"])].append(chunk)

    candidates: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    query_status: list[dict[str, Any]] = []
    for question in questions:
        ranked = rank_question_candidates(question, by_title.get(normalize_alias(question["wikipedia_title"]), []))
        candidate_counts[str(question["question_id"])] = len(ranked)
        candidates.extend(ranked)
        query_status.append(
            {
                "question_id": question["question_id"],
                "wikipedia_title": question["wikipedia_title"],
                "candidate_count": len(ranked),
                "candidate_chunk_ids": [candidate["chunk_id"] for candidate in ranked],
                "status": "candidates_found" if ranked else "no_lexical_candidate",
            }
        )

    output = root / "data/manifests/gold_passage_candidates.jsonl"
    output.write_text("\n".join(json.dumps(candidate, ensure_ascii=False, sort_keys=True) for candidate in candidates) + "\n", encoding="utf-8")
    status_output = root / "data/manifests/gold_candidate_query_status.jsonl"
    status_output.write_text("\n".join(json.dumps(status, ensure_ascii=False, sort_keys=True) for status in query_status) + "\n", encoding="utf-8")
    manifest = {
        "status": "generated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "data/corpus/chunks.jsonl",
        "artifact": "data/manifests/gold_passage_candidates.jsonl",
        "query_status_artifact": "data/manifests/gold_candidate_query_status.jsonl",
        "question_count": len(questions),
        "candidate_count": len(candidates),
        "questions_with_candidates": sum(count > 0 for count in candidate_counts.values()),
        "questions_without_candidates": sum(count == 0 for count in candidate_counts.values()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "ranking": ["alias_occurrences desc", "relation_occurrences desc", "subject_answer_distance asc", "chunk_index asc"],
        "warning": "Candidates are lexical matches only; alias occurrence does not establish relational support.",
    }
    (root / "data/manifests/gold_candidate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
