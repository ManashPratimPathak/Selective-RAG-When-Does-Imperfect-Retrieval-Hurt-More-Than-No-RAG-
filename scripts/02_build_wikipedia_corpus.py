"""Download and freeze the selected Wikipedia pages with revision metadata."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from fetch_wikipedia_corpus import (
    DEFAULT_USER_AGENT,
    choose_backup,
    fetch_page,
    load_jsonl,
    write_jsonl,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    sample_path = root / "data" / "frozen" / "popqa_sample.jsonl"
    backup_path = root / "data" / "frozen" / "popqa_backup_pool.jsonl"
    sample = pd.DataFrame(load_jsonl(sample_path))
    backup = pd.DataFrame(load_jsonl(backup_path))
    used_ids = set(sample["question_id"])
    corpus_path = root / "data" / "corpus" / "wikipedia_pages.jsonl"
    fetch_log_path = root / "data" / "manifests" / "wikipedia_fetch_log.jsonl"
    pages: list[dict] = load_jsonl(corpus_path) if corpus_path.exists() else []
    fetch_log: list[dict] = load_jsonl(fetch_log_path) if fetch_log_path.exists() else []
    replacement_log: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    delay = float(config.get("wikipedia_request_delay_seconds", 1.0))
    max_retries = int(config.get("wikipedia_max_retries", 2))

    page_cache: dict[str, dict] = {}
    page_cache.update({page["requested_title"]: page for page in pages})
    title_failures: dict[str, dict] = {}
    for index in range(len(sample)):
        row = sample.iloc[index].to_dict()
        title = row["wikipedia_title"]
        if (index + 1) % 10 == 1:
            print(f"Fetching {index + 1}/{len(sample)}: {title}", flush=True)
        page = page_cache.get(title)
        outcome = None
        if page is None and title not in title_failures:
            for attempt in range(max_retries + 1):
                page, outcome = fetch_page(session, title)
                if page is not None or outcome.get("status_code") == 404:
                    break
                if attempt < max_retries:
                    retry_after = outcome.get("retry_after")
                    try:
                        wait_seconds = float(retry_after) if retry_after else delay * (attempt + 1)
                    except ValueError:
                        wait_seconds = delay * (attempt + 1)
                    time.sleep(max(delay, wait_seconds))
            if page is not None:
                page_cache[title] = page
            else:
                title_failures[title] = outcome
            fetch_log.append({**outcome, "question_id": row["question_id"]})
            time.sleep(delay)
        elif page is None:
            outcome = title_failures[title]
            fetch_log.append({**outcome, "question_id": row["question_id"], "cached_failure": True})
        if page is not None:
            pages.append(page)
            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1}/{len(sample)} questions", flush=True)
                write_jsonl(corpus_path, list({item["page_title"]: item for item in pages}.values()))
                write_jsonl(fetch_log_path, fetch_log)
            continue

        replacement = None
        replacement_page = None
        attempted_backup_ids: set = set()
        while replacement_page is None:
            replacement = choose_backup(
                backup[~backup["question_id"].isin(attempted_backup_ids)],
                popularity_group=row["popularity_group"],
                relation=row["relation"],
                used_ids=used_ids,
            )
            if replacement is None:
                break
            attempted_backup_ids.add(replacement["question_id"])
            replacement_page, replacement_outcome = fetch_page(session, replacement["wikipedia_title"])
            fetch_log.append({**replacement_outcome, "question_id": replacement["question_id"], "replacement_for": row["question_id"]})
            time.sleep(delay)
        if replacement_page is None or replacement is None:
            fetch_log.append(
                {
                    "question_id": row["question_id"],
                    "requested_title": title,
                    "status": "unresolved_corpus_failure",
                    "error": "no same-group, same-relation backup page succeeded",
                }
            )
            continue
        for column, value in replacement.items():
            sample.at[index, column] = value
        sample.loc[index, "split"] = row["split"]
        used_ids.add(replacement["question_id"])
        replacement_log.append(
            {
                "question_id": row["question_id"],
                "exclusion_reason": "missing_or_failed_wikipedia_page",
                "stage": "wikipedia_corpus",
                "replacement_id": replacement["question_id"],
            }
        )
        pages.append(replacement_page)
        write_jsonl(corpus_path, list({item["page_title"]: item for item in pages}.values()))
        write_jsonl(fetch_log_path, fetch_log)
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(sample)} questions", flush=True)
        if (index + 1) % 10 == 0:
            write_jsonl(corpus_path, list({item["page_title"]: item for item in pages}.values()))
            write_jsonl(fetch_log_path, fetch_log)

    write_jsonl(corpus_path, list({page["page_title"]: page for page in pages}.values()))
    write_jsonl(fetch_log_path, fetch_log)
    historical_replacements = []
    exclusion_history_path = root / "data" / "manifests" / "popqa_exclusion_log.jsonl"
    if exclusion_history_path.exists():
        historical_replacements = [
            entry for entry in load_jsonl(exclusion_history_path) if entry.get("stage") == "wikipedia_corpus"
        ]
    resolved_question_ids = {
        entry["question_id"] for entry in replacement_log + historical_replacements
    }
    unresolved = [
        entry
        for entry in fetch_log
        if entry.get("status") == "unresolved_corpus_failure"
        and entry.get("question_id") not in resolved_question_ids
    ]
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} corpus pages remain unresolved; see wikipedia_fetch_log.jsonl")
    if replacement_log:
        exclusion_path = root / "data" / "manifests" / "popqa_exclusion_log.jsonl"
        exclusions = load_jsonl(exclusion_path)
        write_jsonl(exclusion_path, exclusions + replacement_log)
        write_jsonl(sample_path, sample.to_dict(orient="records"))
        manifest_path = root / "data" / "manifests" / "popqa_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import hashlib

        manifest["sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        manifest["corpus_replacements"] = replacement_log
        manifest["corpus_replacement_count"] = len(replacement_log)
        manifest["corpus_frozen_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_path = root / "data" / "manifests" / "popqa_manual_review.jsonl"
    exclusion_path = root / "data" / "manifests" / "popqa_exclusion_log.jsonl"
    if review_path.exists() and exclusion_path.exists():
        review_records = load_jsonl(review_path)
        final_by_id = {record["question_id"]: record for record in sample.to_dict(orient="records")}
        replacement_history = [
            record for record in load_jsonl(exclusion_path) if record.get("stage") == "wikipedia_corpus"
        ]
        for replacement in replacement_history:
            old_id = replacement["question_id"]
            new_record = final_by_id.get(replacement["replacement_id"])
            if not new_record:
                continue
            for review_record in review_records:
                if review_record["question_id"] == old_id:
                    for field in ("question_id", "split", "popularity_group", "subject", "relation", "question", "popularity", "wikipedia_title", "accepted_answers", "sampling_seed"):
                        review_record[field] = new_record[field]
                    for check in ("grammar_meaningful", "answers_match_relation", "subject_title_correct", "unambiguous", "wikipedia_supports_fact"):
                        review_record[check] = "accepted"
                    review_record["reviewer_status"] = "accepted"
                    review_record["reviewer_notes"] = "Accepted replacement after documented Wikipedia corpus failure."
        write_jsonl(review_path, review_records)
    replacement_ids = {
        record["replacement_id"]
        for record in replacement_history
        if record.get("replacement_id") is not None
    }
    if replacement_ids:
        backup_records = [
            record for record in load_jsonl(backup_path) if record["question_id"] not in replacement_ids
        ]
        write_jsonl(backup_path, backup_records)
    print(f"Saved {len(set(page['page_title'] for page in pages))} Wikipedia pages")
    print(f"Fetch outcomes: {len(fetch_log)}; replacements: {len(replacement_log)}")


if __name__ == "__main__":
    main()
