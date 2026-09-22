"""Fetch and freeze natural PopQA distractor pages.

The default target is 500 pages.  ``--count 1000`` is intended only after the
development retrieval gate says the smaller corpus is too easy; it never uses
test retrieval results to choose the corpus size.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from datasets import Dataset

from distractors import (
    build_distractor_candidates,
    choose_distractor_target,
    excluded_subject_titles,
    sample_distractor_pages,
    selected_relations,
)
from fetch_wikipedia_corpus import DEFAULT_USER_AGENT, fetch_page, load_jsonl, write_jsonl
from prepare_popqa import load_popqa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None, choices=(500, 1000))
    parser.add_argument(
        "--dev-recall-at-3",
        type=float,
        default=None,
        help="Development-only Recall@3 gate; >= configured threshold selects 1,000 pages.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if args.count is not None and args.dev_recall_at_3 is not None:
        parser.error("use either --count or --dev-recall-at-3, not both")
    target = int(args.count or choose_distractor_target(
        args.dev_recall_at_3,
        current_target=int(config.get("distractor_target", 500)),
        expanded_target=int(config.get("distractor_expand_target", 1000)),
        extremely_high_threshold=float(config.get("distractor_dev_recall3_threshold", 0.95)),
    ))
    sample = pd.DataFrame(load_jsonl(root / "data/frozen/popqa_sample.jsonl"))
    backup = pd.DataFrame(load_jsonl(root / "data/frozen/popqa_backup_pool.jsonl"))
    cached_arrows = sorted(Path.home().glob(".cache/huggingface/datasets/**/pop_qa-test.arrow"))
    if cached_arrows:
        source = load_popqa(loader=lambda *_args, **_kwargs: Dataset.from_file(str(cached_arrows[-1])))
    else:
        source = load_popqa()
    candidates = build_distractor_candidates(source, sample, backup=backup, relations=selected_relations(sample))
    excluded_titles = excluded_subject_titles(sample, backup, source)
    selected_head = sample_distractor_pages(
        candidates[candidates["entity_group"] == "head"], count=target // 2, seed=int(config["seed"])
    )
    selected_tail = sample_distractor_pages(
        candidates[candidates["entity_group"] == "tail"], count=target - target // 2, seed=int(config["seed"]) + 1
    )
    selected = pd.concat([selected_head, selected_tail], ignore_index=True)
    manifest_path = root / "data/manifests/distractor_manifest.json"
    distractor_path = root / "data/corpus/distractor_pages.jsonl"
    if args.dry_run:
        print(json.dumps({"target": target, "available": len(candidates), "selected": len(selected)}, indent=2))
        return

    answer_corpus_path = root / "data/corpus/wikipedia_pages.jsonl"
    answer_pages = load_jsonl(answer_corpus_path) if answer_corpus_path.exists() else []
    pages: list[dict] = []
    existing_titles: set[str] = {str(page["page_title"]).casefold() for page in answer_pages}
    existing_ids: set[str] = {str(page["page_id"]) for page in answer_pages}
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    log: list[dict] = []
    delay = float(config.get("wikipedia_request_delay_seconds", 1.0))
    fetched: list[dict] = []
    failed_attempts = 0
    def fetch_candidate(row: pd.Series) -> tuple[dict, dict | None, dict]:
        title = row["page_title"]
        worker_session = requests.Session()
        worker_session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        page = None
        outcome = {}
        for attempt in range(3):
            page, outcome = fetch_page(worker_session, title, timeout=float(config.get("wikipedia_timeout_seconds", 3)))
            if outcome.get("status_code") != 429 or attempt == 2:
                break
            try:
                wait_seconds = float(outcome.get("retry_after") or 31)
            except (TypeError, ValueError):
                wait_seconds = 31
            time.sleep(max(1.0, wait_seconds))
        return row.to_dict(), page, {**outcome, "distractor": True, "entity_group": row["entity_group"], "source_question_id": int(row["source_question_id"])}

    queues = {}
    for offset, group in enumerate(("head", "tail")):
        group_candidates = candidates[candidates["entity_group"] == group]
        chosen_titles = set(selected.loc[selected["entity_group"] == group, "page_title"])
        remainder = group_candidates[~group_candidates["page_title"].isin(chosen_titles)]
        remainder = remainder.sort_values("page_title", kind="mergesort").sample(frac=1, random_state=int(config["seed"]) + offset + 10)
        queues[group] = pd.concat([selected[selected["entity_group"] == group], remainder], ignore_index=True)

    for group, queue in queues.items():
        required_group_count = target // 2 if group == "head" else target - target // 2
        accepted_group_count = 0
        for start in range(0, len(queue), 4):
            if accepted_group_count >= required_group_count:
                break
            batch = [row for _, row in queue.iloc[start : start + 4].iterrows()]
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(fetch_candidate, batch))
            for row_dict, page, outcome in results:
                if accepted_group_count >= required_group_count:
                    break
                title = row_dict["page_title"]
                log.append(outcome)
                if title.casefold() in existing_titles or int(row_dict["source_question_id"]) in set(sample["question_id"]) or title.casefold() in excluded_titles:
                    fetched.append({**row_dict, "status": "excluded_overlap"})
                elif page is not None and page["page_title"].casefold() not in existing_titles and str(page["page_id"]) not in existing_ids:
                    pages.append(page)
                    existing_titles.add(page["page_title"].casefold())
                    existing_ids.add(str(page["page_id"]))
                    accepted_group_count += 1
                    fetched.append({**row_dict, "status": "fetched", "final_page_id": page["page_id"], "final_title": page["page_title"]})
                elif page is not None:
                    failed_attempts += 1
                    fetched.append({**row_dict, "status": "collision_final_page", "final_page_id": page["page_id"], "final_title": page["page_title"]})
                else:
                    failed_attempts += 1
                    fetched.append({**row_dict, "status": "fetch_failed"})
            time.sleep(delay)
            if len(pages) % 10 == 0 and len(pages):
                print(f"Fetched {len(pages)}/{target} unique distractor pages", flush=True)
        if accepted_group_count != required_group_count:
            raise RuntimeError(f"could only freeze {accepted_group_count}/{required_group_count} unique {group} pages")
    if len(pages) != target:
        raise RuntimeError(f"fetched {len(pages)} unique distractor pages; required exactly {target}")
    write_jsonl(distractor_path, pages)
    digest = hashlib.sha256(distractor_path.read_bytes()).hexdigest()
    distractor_ids = [str(page["page_id"]) for page in pages]
    manifest = {
        "corpus_frozen_at": datetime.now(timezone.utc).isoformat(),
        "target_count": target,
        "dev_recall_at_3": args.dev_recall_at_3,
        "dev_gate_threshold": float(config.get("distractor_dev_recall3_threshold", 0.95)),
        "selected_count": len(selected),
        "fetched_count": len(pages),
        "sha256": digest,
        "seed": int(config["seed"]),
        "source": "akariasai/PopQA@latest",
        "relations": sorted(selected_relations(sample)),
        "selection_policy": "unselected PopQA subject pages; s_pop <= 227.0 or s_pop >= 5395.5; frozen sample and backup titles excluded",
        "group_counts": {"head": int(sum(page["entity_group"] == "head" for page in fetched if page["status"] == "fetched")), "tail": int(sum(page["entity_group"] == "tail" for page in fetched if page["status"] == "fetched"))},
        "duplicate_page_ids": len(set(distractor_ids)) != len(distractor_ids),
        "overlap_with_answer_corpus": bool(set(distractor_ids) & {str(page["page_id"]) for page in answer_pages}),
        "replacement_attempts": failed_attempts,
        "unresolved_failures": 0,
        "pages": fetched,
        "fetch_log": log,
        "status": "frozen",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    fetch_log_path = root / "data/manifests/wikipedia_fetch_log.jsonl"
    old_log = load_jsonl(fetch_log_path) if fetch_log_path.exists() else []
    write_jsonl(fetch_log_path, old_log + log)
    print(f"Frozen {manifest['fetched_count']} natural distractor pages (target {target})")


if __name__ == "__main__":
    main()
