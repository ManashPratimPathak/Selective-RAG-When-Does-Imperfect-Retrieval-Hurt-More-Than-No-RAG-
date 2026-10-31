"""Chunk the frozen cleaned page corpus for retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]


def _looks_like_heading(block: str) -> bool:
    words = block.split()
    return len(words) <= 12 and not re.search(r"[.!?]$", block)


def _paragraph_tokens(record: dict[str, Any]) -> tuple[list[str], list[str], list[tuple[str, int]]]:
    blocks = _blocks(record["text"])
    title = record["page_title"]
    section = title
    tokens: list[str] = []
    sections: list[str] = []
    paragraphs: list[tuple[str, int]] = []
    for index, block in enumerate(blocks):
        if index == 0:
            continue
        if _looks_like_heading(block):
            section = block
        paragraph_words = block.split()
        start = len(tokens)
        tokens.extend(paragraph_words)
        sections.extend([section] * len(paragraph_words))
        paragraphs.append((section, len(paragraph_words)))
    return tokens, sections, paragraphs


def chunk_page(
    record: dict[str, Any], *, chunk_words: int = 100, overlap_words: int = 20, min_words: int = 25
) -> list[dict[str, Any]]:
    """Create deterministic overlapping chunks from one cleaned page."""
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be greater than overlap_words")
    tokens, sections, paragraphs = _paragraph_tokens(record)
    if not tokens:
        tokens = record["page_title"].split()
        sections = [record["page_title"]] * len(tokens)
        paragraphs = [(record["page_title"], len(tokens))]

    # A final paragraph shorter than the minimum is folded into the preceding
    # chunk instead of becoming a low-value tail chunk.
    final_short_tail = bool(paragraphs and paragraphs[-1][1] < min_words)
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_words, len(tokens))
        if final_short_tail and end == len(tokens) and start > 0:
            tail = tokens[start + overlap_words :]
            if tail:
                chunks[-1]["text"] = f"{chunks[-1]['text']} {' '.join(tail)}"
                chunks[-1]["word_count"] = len(chunks[-1]["text"].split())
                chunks[-1]["indexed_text"] = f"{record['page_title']}\n\n{chunks[-1]['text']}"
            break
        chunk_tokens = tokens[start:end]
        section_at_content = start + overlap_words if start > 0 else start
        section_title = sections[min(section_at_content, len(sections) - 1)]
        text = " ".join(chunk_tokens)
        chunks.append(_chunk_record(record, index, section_title, text))
        index += 1
        if end == len(tokens):
            break
        start = end - overlap_words

    return chunks


def _chunk_record(record: dict[str, Any], index: int, section_title: str, text: str) -> dict[str, Any]:
    provenance = record.get("provenance", {})
    revision_id = record.get("revision_id") or provenance.get("revision_id")
    source_url = record.get("source_url") or provenance.get("source_url")
    return {
        "chunk_id": f"{record['page_id']}_{revision_id}_{index:04d}",
        "page_id": record["page_id"],
        "revision_id": revision_id,
        "page_title": record["page_title"],
        "requested_title": record.get("requested_title") or record["page_title"],
        "section_title": section_title,
        "chunk_index": index,
        "text": text,
        "indexed_text": f"{record['page_title']}\n\n{text}",
        "word_count": len(text.split()),
        "source_url": source_url,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads("{}")
    config_path = root / "config.yaml"
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    chunk_words = int(config.get("chunk_words", 100))
    overlap_words = int(config.get("chunk_overlap", 20))
    min_words = int(config.get("chunk_min_words", 25))
    source = root / "data/corpus/cleaned_pages.jsonl"
    pages = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [chunk for page in pages for chunk in chunk_page(page, chunk_words=chunk_words, overlap_words=overlap_words, min_words=min_words)]
    output = root / "data/corpus/chunks.jsonl"
    output.write_text("\n".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) for chunk in chunks) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "data/corpus/cleaned_pages.jsonl",
        "artifact": "data/corpus/chunks.jsonl",
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "min_words": min_words,
        "sha256": digest,
    }
    (root / "data/manifests/chunk_corpus_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
