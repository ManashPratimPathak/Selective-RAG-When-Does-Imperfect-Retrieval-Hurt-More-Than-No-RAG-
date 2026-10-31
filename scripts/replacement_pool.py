"""Build a revisioned pool of evidence-backed PopQA replacement candidates."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from .fetch_wikipedia_corpus import fetch_page, load_jsonl, page_has_substantive_content, write_jsonl
    from .prepare_popqa import MALFORMED_RE, MULTI_HOP_RE, TIME_DEPENDENT_RE
    from .clean_wikipedia import clean_page_text
except ImportError:
    from fetch_wikipedia_corpus import fetch_page, load_jsonl, page_has_substantive_content, write_jsonl
    from prepare_popqa import MALFORMED_RE, MULTI_HOP_RE, TIME_DEPENDENT_RE
    from clean_wikipedia import clean_page_text


DATASET_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
CANDIDATE_IDS = [
    1488536, 3111536, 156556, 1409534, 6288361, 1664593, 3093779,
    5924411, 5760400, 2442030, 4783910, 1887458, 2716696,
    6163158, 401885, 1227810, 129760, 906463, 84077, 298575,
    5343981, 985634, 2835997, 4173872, 4939964, 3521531,
    1163970, 5059618,
    2122202, 6261308,
]


def candidate_from_popqa(source, *, q25, q75):
    question = str(source.get("question") or "").strip()
    title = str(source.get("s_wiki_title") or "").strip()
    answer = str(source.get("obj") or "").strip()
    popularity = source.get("s_pop")
    if not question or not title or not answer or popularity is None:
        return None
    if question.count("?") != 1 or TIME_DEPENDENT_RE.search(question) or MULTI_HOP_RE.search(question) or MALFORMED_RE.search(question):
        return None
    if popularity <= q25:
        group = "tail"
    elif popularity >= q75:
        group = "head"
    else:
        return None
    aliases = source.get("o_aliases")
    if isinstance(aliases, str):
        try:
            aliases = json.loads(aliases)
        except ValueError:
            aliases = []
    if not isinstance(aliases, list):
        aliases = []
    accepted = list(dict.fromkeys(
        [answer] + [str(alias).strip() for alias in aliases
                    if str(alias).strip() and not re.search(r"\d{5,}|\bIPI\b", str(alias), re.IGNORECASE)]
    ))
    return {
        "question_id": source["id"],
        "split": "backup",
        "popularity_group": group,
        "subject": source["subj"],
        "relation": source["prop"],
        "question": question,
        "popularity": popularity,
        "wikipedia_title": title,
        "accepted_answers": accepted,
        "sampling_seed": 42,
        "source_obj": answer,
    }


if __name__ == "__main__":
    from datasets import load_dataset

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/manifests/popqa_manifest.json").read_text(encoding="utf-8"))
    q25 = manifest["popularity_thresholds"]["q25"]
    q75 = manifest["popularity_thresholds"]["q75"]
    source = load_dataset("akariasai/PopQA", split="test", revision=DATASET_REVISION, cache_dir=str(root / "data/raw/hf_cache"))
    by_id = {row["id"]: row for row in source if row["id"] in CANDIDATE_IDS}
    used_ids = {row["question_id"] for row in load_jsonl(root / "data/frozen/popqa_sample.jsonl")}
    candidate_path = root / "data/frozen/popqa_replacement_candidates.jsonl"
    page_path = root / "data/corpus/replacement_candidate_pages.jsonl"
    report_path = root / "data/manifests/replacement_candidate_report.jsonl"
    candidates = {row["question_id"]: row for row in load_jsonl(candidate_path)} if candidate_path.exists() else {}
    pages = {page["requested_title"]: page for page in load_jsonl(page_path)} if page_path.exists() else {}
    report = {row["question_id"]: row for row in load_jsonl(report_path)} if report_path.exists() else {}
    with requests.Session() as session:
        for question_id in CANDIDATE_IDS:
            if question_id in used_ids or question_id not in by_id:
                continue
            candidate = candidate_from_popqa(by_id[question_id], q25=q25, q75=q75)
            if candidate is None:
                continue
            title = candidate["wikipedia_title"]
            page = pages.get(title)
            outcome = {"status": "cached"} if page and page_has_substantive_content(page) else None
            if outcome is None:
                page, outcome = fetch_page(session, title)
                time.sleep(0.2)
            if page is None:
                report[question_id] = {"question_id": question_id, "title": title, "status": "fetch_failed", "error": outcome.get("error")}
                continue
            pages[title] = page
            candidates[question_id] = candidate
            clean_text = clean_page_text(page["raw_html"], page_title=page["page_title"], required_texts=candidate["accepted_answers"])
            folded = clean_text.casefold()
            report[question_id] = {
                "question_id": question_id,
                "title": title,
                "page_id": page["page_id"],
                "revision_id": page["revision_id"],
                "cleaned_length": len(clean_text),
                "source_obj": candidate["source_obj"],
                "source_obj_present": candidate["source_obj"].casefold() in folded,
                "preview": clean_text[:600],
                "status": "candidate_fetched",
            }
            write_jsonl(candidate_path, list(candidates.values()))
            write_jsonl(page_path, list(pages.values()))
            write_jsonl(report_path, list(report.values()))
            print(f"{question_id}: {title}: {report[question_id]['status']}", flush=True)
    write_jsonl(report_path, list(report.values()))
    print(json.dumps({"fetched": len(pages), "candidate_count": len(candidates), "dataset_revision": DATASET_REVISION}))
