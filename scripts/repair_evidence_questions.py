"""Replace frozen questions whose current source cannot support their relation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fetch_wikipedia_corpus import choose_backup, load_jsonl, write_jsonl


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/frozen/popqa_sample.jsonl"
    backup_path = root / "data/frozen/popqa_backup_pool.jsonl"
    verification_path = root / "data/manifests/passage_verification_sheet.jsonl"
    exclusion_path = root / "data/manifests/popqa_exclusion_log.jsonl"

    sample = load_jsonl(sample_path)
    backup = pd.DataFrame(load_jsonl(backup_path))
    verification = load_jsonl(verification_path)
    bad_ids = {row["question_id"] for row in verification if row.get("status") != "verified_conservative_protocol"}
    used_ids = {row["question_id"] for row in sample}
    replacements = []
    replaced_ids = set()

    for row in sample:
        if row["question_id"] not in bad_ids:
            continue
        replacement = choose_backup(
            backup[~backup["question_id"].isin(used_ids | replaced_ids)],
            popularity_group=row["popularity_group"],
            relation=row["relation"],
            used_ids=used_ids | replaced_ids,
        )
        if replacement is None:
            continue
        old_id = row["question_id"]
        for field, value in replacement.items():
            if field in {"split", "question_id", "sampling_seed"}:
                continue
            row[field] = value
        row["question_id"] = replacement["question_id"]
        row["sampling_seed"] = 42
        used_ids.add(row["question_id"])
        replaced_ids.add(row["question_id"])
        replacements.append(
            {
                "question_id": old_id,
                "replacement_id": replacement["question_id"],
                "stage": "evidence_verification",
                "exclusion_reason": "source_did_not_support_relation",
            }
        )

    if replacements:
        write_jsonl(sample_path, sample)
        exclusions = load_jsonl(exclusion_path)
        write_jsonl(exclusion_path, exclusions + replacements)
        remaining = [row for row in load_jsonl(backup_path) if row["question_id"] not in replaced_ids]
        write_jsonl(backup_path, remaining)
    print(json.dumps({"replacements": len(replacements), "remaining_unreplaced": len(bad_ids) - len(replacements)}))


if __name__ == "__main__":
    main()
