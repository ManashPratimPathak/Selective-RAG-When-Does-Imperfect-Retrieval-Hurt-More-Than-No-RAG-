"""Tune a retrieval-routing threshold on development data only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .answer_normalization import normalize_answer, normalize_answers
except ImportError:
    from answer_normalization import normalize_answer, normalize_answers


def candidate_thresholds(z_values: list[float]) -> list[float]:
    """Return every distinct routing boundary, including route-none/all bounds."""
    unique = sorted(set(z_values))
    if not unique:
        raise ValueError("development z-scores are required")
    epsilon = 1e-12
    return [unique[0] - epsilon, *unique, unique[-1] + epsilon]


def tune_threshold(
    rows: list[dict[str, Any]],
    z_by_question: dict[int, float],
) -> dict[str, Any]:
    development = [row for row in rows if row.get("split") == "dev"]
    by_question: dict[int, dict[str, bool]] = {}
    for row in development:
        by_question.setdefault(row["question_id"], {})[row["condition"]] = bool(row["exact_match"])
    pairs = [
        (question_id, z_by_question[question_id], values["no_rag"], values["imperfect_rag"])
        for question_id, values in by_question.items()
        if question_id in z_by_question and {"no_rag", "imperfect_rag"} <= values.keys()
    ]
    if not pairs:
        raise ValueError("development No-RAG, Imperfect-RAG, and z-score rows are required")

    candidates = []
    for threshold in candidate_thresholds([z for _, z, _, _ in pairs]):
        routed = [imperfect if z >= threshold else no_rag for _, z, no_rag, imperfect in pairs]
        retrieval_count = sum(z >= threshold for _, z, _, _ in pairs)
        candidates.append({
            "threshold": threshold,
            "em": sum(routed) / len(routed),
            "correct": sum(routed),
            "total": len(routed),
            "retrieval_use": retrieval_count / len(routed),
            "retrieval_count": retrieval_count,
        })
    selected = max(candidates, key=lambda row: (row["em"], -row["retrieval_count"], row["threshold"]))
    return {
        "selection_split": "dev",
        "candidates": candidates,
        "selected": selected,
        "tie_break": "highest development EM, then lower retrieval frequency",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation_path = root / "outputs/step53_dev_validation.jsonl"
    questions_path = root / "data/frozen/popqa_sample.jsonl"
    feature_path = root / "data/retrieval/router_features_standardized.jsonl"
    questions = {
        row["question_id"]: row
        for row in (
            json.loads(line)
            for line in questions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("split") == "dev"
    }
    rows = []
    for line in validation_path.read_text(encoding="utf-8").splitlines():
        validation = json.loads(line)
        question = questions[validation["question_id"]]
        prediction = normalize_answer(validation.get("parsed_answer", ""))
        aliases = normalize_answers(question["accepted_answers"])
        rows.append({
            "question_id": validation["question_id"],
            "split": "dev",
            "condition": validation["condition"],
            "exact_match": bool(prediction) and prediction in {alias for alias in aliases if alias},
        })
    features = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    z_by_question = {row["question_id"]: float(row["s1_z"]) for row in features}
    result = tune_threshold(rows, z_by_question)
    result["development_validation_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    result["development_questions_sha256"] = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    result["standardized_features_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    output = root / "outputs/router_threshold_tuning.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
