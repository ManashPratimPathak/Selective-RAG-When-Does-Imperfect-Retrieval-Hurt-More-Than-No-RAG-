"""Evaluate BM25 Recall@3 on the frozen development split only."""

from __future__ import annotations

import json
from pathlib import Path

from clean_wikipedia import clean_page_text
from rank_bm25 import BM25Okapi
from tokenization import tokenize


def chunks(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step) if words[start : start + size]]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = []
    cleaned_path = root / "data/corpus/cleaned_pages.jsonl"
    if cleaned_path.exists():
        pages = [json.loads(line) for line in cleaned_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        page_texts = ((page["page_id"], page["page_title"], page["text"]) for page in pages)
    else:
        pages = []
        for relative in ("data/corpus/wikipedia_pages.jsonl", "data/corpus/distractor_pages.jsonl"):
            pages.extend(json.loads(line) for line in (root / relative).read_text(encoding="utf-8").splitlines() if line.strip())
        page_texts = ((page["page_id"], page["page_title"], clean_page_text(page["raw_html"], page_title=page["page_title"])) for page in pages)
    for page_id, page_title, text in page_texts:
        for chunk in chunks(text):
            documents.append({"page_id": str(page_id), "page_title": page_title, "text": chunk})

    bm25 = BM25Okapi([tokenize(document["text"]) for document in documents])
    dev = [
        json.loads(line)
        for line in (root / "data/frozen/popqa_sample.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == "dev"
    ]
    hits = 0
    for row in dev:
        scores = bm25.get_scores(tokenize(row["question"]))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:3]
        target = " ".join(row["wikipedia_title"].split()).casefold()
        if any(documents[index]["page_title"].casefold() == target for index in ranked):
            hits += 1
    recall = hits / len(dev) if dev else 0.0
    result = {
        "split": "dev",
        "source_artifact": "data/corpus/cleaned_pages.jsonl" if cleaned_path.exists() else "raw_pages_with_cleaner",
        "questions": len(dev),
        "hits_at_3": hits,
        "recall_at_3": recall,
    }
    (root / "data/manifests/cleaned_recall_at_3.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
