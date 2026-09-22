"""Prepare the PopQA data split and write its manifest."""

from pathlib import Path

import yaml
import json
import pandas as pd

from prepare_popqa import (
    apply_eligibility_checks,
    build_backup_pool,
    build_frozen_frame,
    build_manual_review_ledger,
    compute_popularity_thresholds,
    freeze_sample,
    label_popularity_groups,
    load_popqa,
    relation_stratified_sample,
    replace_rejected_rows,
    review_frozen_frame,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    manifest_path = root / "data" / "manifests" / "popqa_manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    pre_review_sha256 = previous_manifest.get("pre_review_sha256") or previous_manifest.get("sha256")
    frame = load_popqa()
    eligible, exclusions = apply_eligibility_checks(frame)
    thresholds = compute_popularity_thresholds(eligible)
    grouped = label_popularity_groups(eligible, thresholds)
    sample = relation_stratified_sample(
        grouped,
        head_size=int(config["head_dev"] + config["head_test"]),
        tail_size=int(config["tail_dev"] + config["tail_test"]),
        dev_size=int(config["head_dev"]),
        seed=int(config["seed"]),
    )
    backup_pool = build_backup_pool(grouped, sample, per_group=20, seed=int(config["seed"]))
    frozen = build_frozen_frame(sample, sampling_seed=int(config["seed"]))
    review_ledger = review_frozen_frame(frozen, frame)
    frozen, manual_replacements = replace_rejected_rows(frozen, review_ledger, backup_pool)
    if len(manual_replacements):
        review_ledger = review_frozen_frame(frozen, frame)
    exclusions = pd.concat([exclusions, manual_replacements], ignore_index=True)
    relation_distribution = {
        str(group): {str(relation): int(count) for relation, count in subset["relation"].value_counts().items()}
        for group, subset in frozen.groupby("popularity_group")
    }
    backup_frozen = build_frozen_frame(backup_pool, sampling_seed=int(config["seed"]))
    backup_path = root / "data" / "frozen" / "popqa_backup_pool.jsonl"
    backup_frozen.to_json(backup_path, orient="records", lines=True, force_ascii=False)
    if review_ledger["reviewer_status"].isin(["pending_manual_review", "rejected"]).any():
        raise RuntimeError("manual review is incomplete; refusing to freeze the final dataset")
    freeze_sample(
        frozen,
        output_path=root / "data" / "frozen" / "popqa_sample.jsonl",
        manifest_path=manifest_path,
        source="akariasai/PopQA@latest",
        config=config,
        exclusion_log=exclusions,
        exclusion_path=root / "data" / "manifests" / "popqa_exclusion_log.jsonl",
        popularity_thresholds=thresholds,
        relation_distribution=relation_distribution,
        manual_review_ledger=review_ledger,
        manual_review_path=root / "data" / "manifests" / "popqa_manual_review.jsonl",
        pre_review_sha256=pre_review_sha256,
        backup_pool_path=backup_path,
        backup_pool_count={str(group): int(count) for group, count in backup_pool.groupby("popularity_group").size().items()},
    )
    print(f"Frozen {len(sample)} PopQA rows")


if __name__ == "__main__":
    main()
