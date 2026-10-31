"""Evaluate fixed retrieval policies on the frozen test set."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def nearest_rank_percentile(values: list[float], percentile: int) -> float:
    index = max(0, (len(values) * percentile + 99) // 100 - 1)
    return sorted(values)[index]


def evaluate_policy(
    name: str,
    rows: list[dict[str, Any]],
    latencies: dict[tuple[int, str], float],
    selected_condition: dict[int, str],
) -> dict[str, Any]:
    by_question: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = row
    selected = [by_question[q][condition] for q, condition in selected_condition.items()]
    baseline = [by_question[q]["no_rag"] for q in selected_condition]
    chosen_conditions = list(selected_condition.values())
    damage = sum(base["exact_match"] and not current["exact_match"] for base, current in zip(baseline, selected))
    rescue = sum(not base["exact_match"] and current["exact_match"] for base, current in zip(baseline, selected))
    baseline_correct = sum(row["exact_match"] for row in baseline)
    baseline_incorrect = len(baseline) - baseline_correct
    selected_latencies = [latencies[(q, condition)] for q, condition in selected_condition.items()]
    return {
        "em": sum(row["exact_match"] for row in selected) / len(selected),
        "token_f1": sum(row["token_f1"] for row in selected) / len(selected),
        "damage_rate": damage / baseline_correct if baseline_correct else 0.0,
        "rescue_rate": rescue / baseline_incorrect if baseline_incorrect else 0.0,
        "routed_to_rag": sum(condition != "no_rag" for condition in chosen_conditions) / len(chosen_conditions),
        "latency_ms": {
            "median": statistics.median(selected_latencies),
            "p95": nearest_rank_percentile(selected_latencies, 95),
        },
        "question_count": len(selected),
        "damage": damage,
        "rescue": rescue,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "outputs/test_scoring_ledger.jsonl"
    latency_path = root / "outputs/test_latency.jsonl"
    features_path = root / "data/retrieval/router_features_standardized.jsonl"
    tuning_path = root / "outputs/router_threshold_tuning.json"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latencies = {
        (row["question_id"], row["condition"]): float(row["total_latency_ms"])
        for row in (json.loads(line) for line in latency_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    test_z = {
        row["question_id"]: float(row["s1_z"])
        for row in (json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if row.get("split") == "test"
    }
    threshold = float(json.loads(tuning_path.read_text(encoding="utf-8"))["selected"]["threshold"])
    question_ids = sorted(test_z)
    policies = {
        "never_retrieve": {question_id: "no_rag" for question_id in question_ids},
        "always_retrieve": {question_id: "imperfect_rag" for question_id in question_ids},
        "threshold_router": {
            question_id: "imperfect_rag" if z >= threshold else "no_rag"
            for question_id, z in test_z.items()
        },
        "clean_rag": {question_id: "clean_rag" for question_id in question_ids},
    }
    results = {
        name: evaluate_policy(name, rows, latencies, selected)
        for name, selected in policies.items()
    }
    output = root / "outputs/policy_evaluation.json"
    output.write_text(
        json.dumps(
            {
                "test_only": True,
                "threshold": threshold,
                "threshold_source": "development-selected fixed threshold",
                "policies": results,
                "inputs": {
                    "scoring_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                    "latency_sha256": hashlib.sha256(latency_path.read_bytes()).hexdigest(),
                    "standardized_features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
