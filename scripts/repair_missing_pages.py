"""Fetch active sample pages absent from the frozen corpus cache."""

from pathlib import Path

import requests

from fetch_wikipedia_corpus import DEFAULT_USER_AGENT, fetch_page, load_jsonl, page_has_substantive_content, write_jsonl


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    corpus_path = root / "data/corpus/wikipedia_pages.jsonl"
    log_path = root / "data/manifests/wikipedia_fetch_log.jsonl"
    sample = load_jsonl(sample_path)
    pages = load_jsonl(corpus_path)
    active_titles = {row["wikipedia_title"] for row in sample}
    present = {page.get("requested_title") for page in pages}
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    log = load_jsonl(log_path)
    for title in sorted(active_titles - present):
        page, outcome = fetch_page(session, title)
        log.append({**outcome, "repair_missing_page": True})
        if page is not None and page_has_substantive_content(page):
            pages.append(page)
    write_jsonl(corpus_path, list({page["page_title"]: page for page in pages}.values()))
    write_jsonl(log_path, log)
    print(f"active_missing_before={len(active_titles - present)} active_pages={len(pages)}")


if __name__ == "__main__":
    main()
