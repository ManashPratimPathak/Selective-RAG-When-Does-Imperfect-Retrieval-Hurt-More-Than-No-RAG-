"""Create a derived scoring ledger without changing frozen predictions."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .answer_normalization import normalize_answer, normalize_answers
except ImportError:
    from answer_normalization import normalize_answer, normalize_answers

FROZEN_PREDICTIONS_SHA256 = "80e8b1376677a05266ef233fa12fc3f75c4e8347a18fbd0a5106d3b6ae48b45a"
NORMALIZATION_SPEC = {
    "case": "lowercase",
    "punctuation": "remove Unicode punctuation characters",
    "articles": ["a", "an", "the"],
    "article_scope": "whole whitespace-delimited tokens only",
    "whitespace": "collapse runs and strip surrounding whitespace",
    "match_rule": "match any normalized accepted alias using full-string equality",
}


def normalization_spec_hash() -> str:
    payload = json.dumps(NORMALIZATION_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_normalization_spec(path: Path) -> dict[str, Any]:
    """Write or validate the immutable normalization contract."""
    frozen = {**NORMALIZATION_SPEC, "spec_hash": normalization_spec_hash()}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("normalization specification has changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_prediction(raw_prediction: object, raw_aliases: list[object]) -> dict[str, Any]:
    """Return raw and normalized values while scoring normalized exact match."""
    normalized_prediction = normalize_answer(raw_prediction)
    normalized_aliases = normalize_answers(raw_aliases)
    return {
        "raw_prediction": raw_prediction,
        "normalized_prediction": normalized_prediction,
        "raw_aliases": raw_aliases,
        "normalized_aliases": normalized_aliases,
        "exact_match": alias_aware_exact_match(normalized_prediction, normalized_aliases),
        "token_f1": max_token_f1(normalized_prediction, normalized_aliases),
    }


def alias_aware_exact_match(normalized_prediction: str, normalized_aliases: list[str]) -> bool:
    """Match only when the complete normalized prediction equals an alias."""
    return bool(normalized_prediction) and any(
        normalized_prediction == alias for alias in normalized_aliases if alias
    )


def token_f1(normalized_prediction: str, normalized_alias: str) -> float:
    """Calculate token F1 using multiset overlap."""
    prediction_tokens = normalized_prediction.split()
    alias_tokens = normalized_alias.split()
    if not prediction_tokens or not alias_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(alias_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(alias_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(normalized_prediction: str, normalized_aliases: list[str]) -> float:
    return max((token_f1(normalized_prediction, alias) for alias in normalized_aliases), default=0.0)


def build_scoring_row(prediction: dict[str, Any], question: dict[str, Any], spec_hash: str) -> dict[str, Any]:
    scored = score_prediction(prediction["raw_prediction"], question["accepted_answers"])
    return {
        "question_id": prediction["question_id"],
        "split": question["split"],
        "popularity_group": question["popularity_group"],
        "condition": prediction["condition"],
        "raw_prediction": scored["raw_prediction"],
        "normalized_prediction": scored["normalized_prediction"],
        "raw_accepted_aliases": scored["raw_aliases"],
        "normalized_accepted_aliases": scored["normalized_aliases"],
        "exact_match": scored["exact_match"],
        "token_f1": scored["token_f1"],
        "normalization_spec_hash": spec_hash,
    }


def write_scoring_ledger(
    predictions_path: Path,
    questions: Iterable[dict[str, Any]],
    ledger_path: Path,
    spec_path: Path,
) -> None:
    if _file_sha256(predictions_path) != FROZEN_PREDICTIONS_SHA256:
        raise ValueError("frozen prediction artifact hash does not match the required digest")
    spec = freeze_normalization_spec(spec_path)
    by_id = {row["question_id"]: row for row in questions}
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [build_scoring_row(row, by_id[row["question_id"]], spec["spec_hash"]) for row in predictions]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    totals: dict[str, list[bool]] = {}
    for row in rows:
        totals.setdefault(row["condition"], []).append(row["exact_match"])
    accuracy = {
        condition: {
            "correct": sum(matches),
            "total": len(matches),
            "accuracy": sum(matches) / len(matches) if matches else 0.0,
        }
        for condition, matches in sorted(totals.items())
    }
    damage = damage_rate(rows)
    rescue = rescue_rate(rows)
    change = accuracy_change(accuracy)
    paired = paired_statistics(rows)
    (ledger_path.parent / "test_scoring_accuracy.json").write_text(
        json.dumps(
            {
                "normalization_spec_hash": spec["spec_hash"],
                "conditions": accuracy,
                "damage_rate": damage,
                "rescue_rate": rescue,
                "accuracy_change": change,
                "transition_table": transition_table(rows),
                "paired_statistics": paired,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def damage_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure imperfect-retrieval failures among No-RAG-correct questions."""
    by_question: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    no_rag_correct = sum(values.get("no_rag", False) for values in by_question.values())
    damaged = sum(
        values.get("no_rag", False) and not values.get("imperfect_rag", False)
        for values in by_question.values()
    )
    return {
        "damaged": damaged,
        "baseline_correct": no_rag_correct,
        "rate": damaged / no_rag_correct if no_rag_correct else 0.0,
    }


def rescue_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure imperfect-retrieval corrections among No-RAG failures."""
    by_question: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    no_rag_incorrect = sum(not values.get("no_rag", False) for values in by_question.values())
    rescued = sum(
        not values.get("no_rag", False) and values.get("imperfect_rag", False)
        for values in by_question.values()
    )
    return {
        "rescued": rescued,
        "baseline_incorrect": no_rag_incorrect,
        "rate": rescued / no_rag_incorrect if no_rag_incorrect else 0.0,
    }


def accuracy_change(accuracy: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return Imperfect-RAG minus No-RAG accuracy in percentage points."""
    no_rag = float(accuracy["no_rag"]["accuracy"])
    imperfect = float(accuracy["imperfect_rag"]["accuracy"])
    return {
        "imperfect_rag_minus_no_rag": imperfect - no_rag,
        "percentage_points": (imperfect - no_rag) * 100,
    }


def transition_table(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count paired No-RAG/Imperfect-RAG correctness transitions."""
    by_question: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    table = {
        "stable_correct": 0,
        "damage": 0,
        "rescue": 0,
        "stable_failure": 0,
    }
    for values in by_question.values():
        no_rag = values.get("no_rag", False)
        imperfect = values.get("imperfect_rag", False)
        if no_rag and imperfect:
            table["stable_correct"] += 1
        elif no_rag:
            table["damage"] += 1
        elif imperfect:
            table["rescue"] += 1
        else:
            table["stable_failure"] += 1
    return table


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_statistics(rows: list[dict[str, Any]], *, samples: int = 20_000, seed: int = 0) -> dict[str, Any]:
    """Return a paired-bootstrap CI and exact two-sided McNemar test."""
    by_question: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    pairs = [
        (values.get("no_rag", False), values.get("imperfect_rag", False))
        for values in by_question.values()
        if "no_rag" in values and "imperfect_rag" in values
    ]
    if not pairs:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "mcnemar_p_value": 1.0, "significant_at_0_05": False}
    differences = [int(imperfect) - int(no_rag) for no_rag, imperfect in pairs]
    delta = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap = [
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    ]
    bootstrap.sort()
    damage = sum(no_rag and not imperfect for no_rag, imperfect in pairs)
    rescue = sum(not no_rag and imperfect for no_rag, imperfect in pairs)
    discordant = damage + rescue
    lower_tail = sum(math.comb(discordant, i) for i in range(min(damage, rescue) + 1)) / (2**discordant) if discordant else 1.0
    p_value = min(1.0, 2 * lower_tail)
    return {
        "delta": delta,
        "ci95": [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)],
        "ci_method": "paired bootstrap percentile, 20000 resamples, seed 0",
        "mcnemar_damage": damage,
        "mcnemar_rescue": rescue,
        "mcnemar_p_value": p_value,
        "significant_at_0_05": p_value < 0.05,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    questions = [
        json.loads(line)
        for line in (root / "data/frozen/popqa_sample.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    write_scoring_ledger(
        root / "outputs/test_predictions.jsonl",
        questions,
        root / "outputs/test_scoring_ledger.jsonl",
        root / "data/manifests/answer_normalization_spec.json",
    )


if __name__ == "__main__":
    main()
