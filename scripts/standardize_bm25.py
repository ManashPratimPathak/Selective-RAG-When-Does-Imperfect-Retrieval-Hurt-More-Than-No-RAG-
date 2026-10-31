"""Standardize top BM25 scores using development data only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def fit_dev_standardizer(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    scores = [float(row["s1"]) for row in rows if row.get("split") == "dev"]
    if not scores:
        raise ValueError("development rows are required")
    standard_deviation = pstdev(scores)
    if standard_deviation == 0:
        raise ValueError("development top BM25 scores have zero standard deviation")
    return {"dev_count": len(scores), "dev_mean": mean(scores), "dev_std": standard_deviation}


def apply_standardizer(rows: list[dict[str, Any]], stats: dict[str, float | int]) -> list[dict[str, Any]]:
    mean_score = float(stats["dev_mean"])
    standard_deviation = float(stats["dev_std"])
    return [
        {**row, "s1_z": (float(row["s1"]) - mean_score) / standard_deviation}
        for row in rows
    ]


def standardize_file(source_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = fit_dev_standardizer(rows)
    standardized = apply_standardizer(rows, stats)
    development_ids = sorted(row["question_id"] for row in rows if row.get("split") == "dev")
    development_z = [row["s1_z"] for row in standardized if row.get("split") == "dev"]
    development_ids_payload = json.dumps(development_ids, separators=(",", ":"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in standardized),
        encoding="utf-8",
    )
    manifest = {
        "source_artifact": str(source_path),
        "standardized_artifact": str(output_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "standardized_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "score_field": "s1",
        "standardized_field": "s1_z",
        "formula": "s1_z = (s1 - dev_mean) / dev_std",
        "standard_deviation_ddof": 0,
        "development_question_ids": development_ids,
        "development_ids_sha256": hashlib.sha256(development_ids_payload.encode("utf-8")).hexdigest(),
        "development_standardized_mean": mean(development_z),
        "raw_s1_preserved": True,
        "statistics": stats,
        "test_application": "apply development mean and standard deviation unchanged",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = standardize_file(
        root / "data/retrieval/router_features_top5.jsonl",
        root / "data/retrieval/router_features_standardized.jsonl",
        root / "data/manifests/bm25_standardization.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
