"""Paired McNemar and bootstrap analysis for No-RAG vs Imperfect-RAG."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260929


def paired_rows(rows: list[dict[str, Any]], group: str | None = None) -> list[tuple[bool, bool]]:
    by_question: dict[int, dict[str, Any]] = {}
    for row in rows:
        if group is not None and row.get("popularity_group") != group:
            continue
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    return [
        (values["no_rag"], values["imperfect_rag"])
        for values in by_question.values()
        if {"no_rag", "imperfect_rag"} <= values.keys()
    ]


def transition_counts(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    return {
        "a_stable_correct": sum(no and imperfect for no, imperfect in pairs),
        "b_damage": sum(no and not imperfect for no, imperfect in pairs),
        "c_rescue": sum(not no and imperfect for no, imperfect in pairs),
        "d_stable_failure": sum(not no and not imperfect for no, imperfect in pairs),
    }


def _delta(no: np.ndarray, imperfect: np.ndarray) -> float:
    return float(np.mean(imperfect.astype(float) - no.astype(float)))


def _damage(no: np.ndarray, imperfect: np.ndarray) -> float:
    denominator = int(no.sum())
    return float(np.logical_and(no, ~imperfect).sum() / denominator) if denominator else 0.0


def _rescue(no: np.ndarray, imperfect: np.ndarray) -> float:
    denominator = int((~no).sum())
    return float(np.logical_and(~no, imperfect).sum() / denominator) if denominator else 0.0


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    if not trials:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return [center - half_width, center + half_width]


def _bootstrap_ci(pairs: list[tuple[bool, bool]], statistic: Any, rng: np.random.Generator) -> list[float]:
    no = np.array([pair[0] for pair in pairs], dtype=bool)
    imperfect = np.array([pair[1] for pair in pairs], dtype=bool)
    result = bootstrap(
        (no, imperfect), statistic, paired=True, vectorized=False,
        n_resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95,
        method="percentile", rng=rng,
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def analyze_group(pairs: list[tuple[bool, bool]], rng: np.random.Generator) -> dict[str, Any]:
    counts = transition_counts(pairs)
    table = [[counts["a_stable_correct"], counts["b_damage"]], [counts["c_rescue"], counts["d_stable_failure"]]]
    test = mcnemar(table, exact=True)
    no = np.array([pair[0] for pair in pairs], dtype=bool)
    imperfect = np.array([pair[1] for pair in pairs], dtype=bool)
    damage_trials = counts["a_stable_correct"] + counts["b_damage"]
    rescue_trials = counts["c_rescue"] + counts["d_stable_failure"]
    return {
        "n": len(pairs),
        "transition_table": counts,
        "mcnemar": {"method": "exact", "statistic": float(test.statistic), "p_value": float(test.pvalue)},
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "delta_em": {"estimate": _delta(no, imperfect), "ci95": _bootstrap_ci(pairs, _delta, rng)},
            "damage_rate": {
                "estimate": _damage(no, imperfect),
                "ci95": wilson_interval(counts["b_damage"], damage_trials),
                "ci_method": "Wilson score interval, conditional on No-RAG correct",
            },
            "rescue_rate": {
                "estimate": _rescue(no, imperfect),
                "ci95": wilson_interval(counts["c_rescue"], rescue_trials),
                "ci_method": "Wilson score interval, conditional on No-RAG incorrect",
            },
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "outputs/test_scoring_ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = {
        "split": "test",
        "comparison": "No-RAG vs Imperfect-RAG",
        "canonical_interval_methods": {
            "delta_em": "SciPy paired bootstrap percentile interval",
            "conditional_damage_and_rescue_rates": "Wilson score interval",
        },
        "superseded_results": [
            {
                "metric": "overall_delta_em_ci95",
                "ci95": [0.405, 0.56],
                "reason": "superseded by the canonical SciPy paired percentile result below",
            }
        ],
        "groups": {
            group: analyze_group(paired_rows(rows, group), rng)
            for group in (None, "head", "tail")
        },
        "inputs": {"scoring_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest()},
    }
    result["groups"]["overall"] = result["groups"].pop(None)
    output = root / "outputs/paired_statistics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "outputs/paired_statistics.manifest.json").write_text(
        json.dumps(
            {
                "artifact": str(output),
                "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "split": "test",
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
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
