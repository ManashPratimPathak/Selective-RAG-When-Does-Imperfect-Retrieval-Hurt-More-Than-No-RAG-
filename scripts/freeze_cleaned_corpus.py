"""Clean, validate, and freeze the full page-level retrieval corpus."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clean_wikipedia import clean_page_text


def clean_page_pair(raw_html, page_title, role, required_texts, *, cleaner=clean_page_text):
    text = cleaner(raw_html, page_title=page_title, required_texts=required_texts)
    baseline = cleaner(raw_html, page_title=page_title) if role == "gold" else None
    return text, baseline


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().casefold()


def accepted_answers(row: dict[str, Any]) -> list[str]:
    values = row.get("accepted_answers", [])
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if str(value).strip()]


def answer_fragments(answer: str) -> list[str]:
    """Return exact and conservative entity-name variants for evidence checks."""
    fragments = [answer]
    tokens = re.findall(r"[\w]+", answer, flags=re.UNICODE)
    if len(tokens) >= 2:
        fragments.append(" ".join(tokens[:2]))
        fragments.append(" ".join(tokens[-2:]))
    return fragments


def answer_supported(text: str, answer: str) -> bool:
    """Allow normal inflection/context between tokens of a short answer."""
    text_tokens = re.findall(r"[\w]+", normalise(text), flags=re.UNICODE)
    stopwords = {"a", "an", "the", "of", "in", "on", "for", "and"}
    for candidate in answer_fragments(answer):
        answer_tokens = [
            token for token in re.findall(r"[\w]+", normalise(candidate), flags=re.UNICODE) if token not in stopwords
        ]
        if not answer_tokens:
            continue
        if normalise(candidate) in normalise(text):
            return True
        try:
            first = text_tokens.index(answer_tokens[0])
            cursor = first + 1
            for token in answer_tokens[1:]:
                cursor = text_tokens.index(token, cursor) + 1
            if cursor - first <= len(answer_tokens) + 3:
                return True
        except ValueError:
            continue
    return False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = load_jsonl(root / "data/corpus/wikipedia_pages.jsonl")
    distractors = load_jsonl(root / "data/corpus/distractor_pages.jsonl")
    sample_by_title: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(root / "data/frozen/popqa_sample.jsonl"):
        sample_by_title.setdefault(normalise(row["wikipedia_title"]), []).append(row)
    distractor_manifest = json.loads((root / "data/manifests/distractor_manifest.json").read_text(encoding="utf-8"))
    distractor_info = {
        normalise(row["page_title"]): row
        for row in distractor_manifest.get("pages", [])
        if row.get("status") == "fetched"
    }

    source_pages: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    source_pages.extend(
        (page, "gold", sample_by_title.get(normalise(page.get("requested_title") or page["page_title"]), []))
        for page in gold
    )
    source_pages.extend((page, "distractor", distractor_info.get(normalise(page["page_title"]))) for page in distractors)

    if len(source_pages) != len(gold) + len(distractors):
        raise RuntimeError("source page count changed during corpus assembly")
    matched_question_count = sum(len(extra) for _, role, extra in source_pages if role == "gold")
    question_count = sum(len(rows) for rows in sample_by_title.values())
    if matched_question_count != question_count:
        raise RuntimeError(f"only {matched_question_count}/{question_count} questions map to frozen gold pages")
    page_ids = [str(page["page_id"]) for page, _, _ in source_pages]
    if len(set(page_ids)) != len(page_ids):
        raise RuntimeError("source page IDs are not unique")

    cleaned: list[dict[str, Any]] = []
    answer_checks: list[dict[str, Any]] = []
    for page, role, extra in source_pages:
        answers = [answer for record in extra for answer in accepted_answers(record)] if role == "gold" else []
        preservation_fragments = [fragment for answer in answers for fragment in answer_fragments(answer)]
        text, baseline_text = clean_page_pair(
            page["raw_html"], page["page_title"], role, preservation_fragments
        )
        if not text.strip():
            raise RuntimeError(f"empty cleaned document: {page['page_id']}")
        if re.search(r"</?[A-Za-z][^>]*>", text):
            raise RuntimeError(f"HTML-like text remains: {page['page_id']}")
        if re.search(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]", text):
            raise RuntimeError(f"citation marker remains: {page['page_id']}")

        metadata = {
            "page_id": str(page["page_id"]),
            "page_title": page["page_title"],
            "requested_title": page.get("requested_title") or page["page_title"],
            "role": role,
            "provenance": {
                key: page.get(key)
                for key in ("requested_title", "source_url", "revision_id", "revision_timestamp", "retrieved_at", "license")
                if page.get(key) is not None
            },
            "text": text,
        }
        if role == "distractor":
            metadata["distractor_group"] = (extra or {}).get("entity_group")
            metadata["source_question_id"] = (extra or {}).get("source_question_id")
        cleaned.append(metadata)

        if role == "gold":
            matched = [answer for answer in answers if answer_supported(text, answer)]
            source_matched = [answer for answer in answers if answer_supported(baseline_text, answer)]
            if source_matched and not matched:
                raise RuntimeError(f"accepted-answer evidence removed from page {page['page_id']}")
            answer_checks.append(
                {
                    "page_id": str(page["page_id"]),
                    "accepted_answers": answers,
                    "matched_answers": matched,
                    "source_evidence_status": "preserved" if matched else "not_present_in_source_text",
                }
            )

    output_path = root / "data/corpus/cleaned_pages.jsonl"
    write_jsonl(output_path, cleaned)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    sample_rows = []
    for row in cleaned[:: max(1, len(cleaned) // 10)][:10]:
        sample_rows.append({"page_id": row["page_id"], "page_title": row["page_title"], "role": row["role"], "preview": row["text"][:500]})
    write_jsonl(root / "data/manifests/cleaned_corpus_manual_sample.jsonl", sample_rows)
    (root / "data/manifests/cleaned_answer_checks.json").write_text(
        json.dumps(answer_checks, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": {"gold": len(gold), "distractor": len(distractors), "total": len(cleaned)},
        "cleaned_document_count": len(cleaned),
        "sha256": digest,
        "artifact": "data/corpus/cleaned_pages.jsonl",
        "contains_raw_html": False,
        "page_ids_unique": True,
        "empty_documents": 0,
        "accepted_answer_check": "passed: preserved when present; source absence recorded separately",
        "short_fragment_policy": "20-character minimum for ordinary paragraphs; accepted-answer matches are retained regardless of length",
        "manual_sample": "data/manifests/cleaned_corpus_manual_sample.jsonl",
    }
    (root / "data/manifests/cleaned_corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
