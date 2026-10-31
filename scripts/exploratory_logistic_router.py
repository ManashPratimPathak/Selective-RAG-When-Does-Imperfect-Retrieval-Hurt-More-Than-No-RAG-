"""Exploratory learned router; not a production or test-selected policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .answer_normalization import normalize_answer, normalize_answers
except ImportError:
    from answer_normalization import normalize_answer, normalize_answers

FEATURE_NAMES = ("s1", "score_margin", "query_top_passage_token_overlap", "question_token_count")


def build_dev_dataset(
    features: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    questions: dict[int, dict[str, Any]],
) -> tuple[list[list[float]], list[int]]:
    feature_by_id = {row["question_id"]: row for row in features if row.get("split") == "dev"}
    correctness: dict[int, dict[str, bool]] = {}
    for row in validation:
        question = questions[row["question_id"]]
        prediction = normalize_answer(row.get("parsed_answer", ""))
        aliases = normalize_answers(question["accepted_answers"])
        correctness.setdefault(row["question_id"], {})[row["condition"]] = bool(prediction) and prediction in {
            alias for alias in aliases if alias
        }
    x: list[list[float]] = []
    y: list[int] = []
    for question_id, scores in feature_by_id.items():
        outcomes = correctness.get(question_id, {})
        if {"no_rag", "imperfect_rag"} <= outcomes.keys():
            x.append([float(scores[name]) for name in FEATURE_NAMES])
            y.append(int(not outcomes["no_rag"] and outcomes["imperfect_rag"]))
    if len(set(y)) < 2:
        raise ValueError("exploratory router needs both rescue and non-rescue labels")
    return x, y


def run_exploratory_router(x: list[list[float]], y: list[int]) -> dict[str, Any]:
    pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipeline, x, y, cv=folds, scoring="accuracy")
    pipeline.fit(x, y)
    model = pipeline.named_steps["logistic_regression"]
    return {
        "feature_names": list(FEATURE_NAMES),
        "target_label": "positive=No-RAG incorrect and Imperfect-RAG correct (rescue); negative=all other paired outcomes",
        "class_counts": {"negative": y.count(0), "positive": y.count(1)},
        "majority_class_baseline_accuracy": max(y.count(0), y.count(1)) / len(y),
        "five_fold_cv_accuracy_mean": float(scores.mean()),
        "five_fold_cv_accuracy_std": float(scores.std()),
        "five_fold_scores": scores.tolist(),
        "cross_validation": {"method": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 0},
        "coefficients_on_scaled_features": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "note": "Exploratory only: 40 development examples are too few for a strong learned-router claim.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data/retrieval/router_features_standardized.jsonl"
    validation_path = root / "outputs/step53_dev_validation.jsonl"
    questions_path = root / "data/frozen/popqa_sample.jsonl"
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = {row["question_id"]: row for row in read(questions_path) if row.get("split") == "dev"}
    x, y = build_dev_dataset(read(features_path), read(validation_path), questions)
    result = run_exploratory_router(x, y)
    result.update({
        "analysis": "exploratory",
        "split": "dev",
        "example_count": len(y),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    })
    output = root / "outputs/exploratory_logistic_router.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "analysis": "exploratory",
        "split": "dev",
    }
    (root / "outputs/exploratory_logistic_router.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
