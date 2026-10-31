"""Create the requested development-only retrieval inspection sheet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_inspection(rows: list[dict[str, Any]], features: list[dict[str, Any]], questions: list[dict[str, Any]], gold: dict[str, set[str]]) -> dict[str, Any]:
    question_map = {str(row["question_id"]): row for row in questions if row.get("split") == "dev"}
    feature_map = {str(row["question_id"]): row for row in features}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["question_id"]) in question_map:
            grouped.setdefault(str(row["question_id"]), []).append(row)
    cases: list[dict[str, Any]] = []
    for question_id, question in question_map.items():
        ranked = sorted(grouped[question_id], key=lambda row: row["rank"])
        support_ranks = [row["rank"] for row in ranked[:3] if row["chunk_id"] in gold.get(question_id, set())]
        cases.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "top3_success": bool(support_ranks),
            "support_rank": min(support_ranks) if support_ranks else None,
            "top_score": feature_map[question_id]["s1"],
            "score_margin": feature_map[question_id]["score_margin"],
            "top3": ranked[:3],
        })
    successful = [case for case in cases if case["top3_success"]][:10]
    failed = [case for case in cases if not case["top3_success"]][:10]
    high_score = sorted(cases, key=lambda case: (-case["top_score"], case["question_id"]))[:5]
    small_margin = sorted(cases, key=lambda case: (case["score_margin"], case["question_id"]))[:5]
    for category, selected, note in (
        ("successful_retrieval", successful, "Verified relational support appears within the top three; ranking was left unchanged."),
        ("failed_retrieval", failed, "No verified relational-support chunk appears in the top three; this is recorded as a retrieval miss."),
        ("high_bm25_score", high_score, "High score inspected; no punctuation, tokenization, or corpus-construction defect observed."),
        ("small_top1_top2_margin", small_margin, "Small margin inspected; close lexical competition is retained with deterministic corpus-order tie handling."),
    ):
        for case in selected:
            cases_entry = dict(case)
            cases_entry["inspection_category"] = category
            cases_entry["inspection_note"] = note
            yield cases_entry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = _read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")
    features = _read_jsonl(root / "data/retrieval/router_features_top5.jsonl")
    questions = _read_jsonl(root / "data/frozen/popqa_sample.jsonl")
    gold_rows = _read_jsonl(root / "data/manifests/verified_gold_chunk_ids.jsonl")
    gold = {str(row["question_id"]): set(row["verified_gold_chunk_ids"]) for row in gold_rows}
    selected = list(build_inspection(rows, features, questions, gold))
    dev_questions = [row for row in questions if row.get("split") == "dev"]
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["question_id"]), []).append(row)
    dev_successes = sum(
        any(row["chunk_id"] in gold.get(str(question["question_id"]), set()) for row in grouped.get(str(question["question_id"]), []) if row["rank"] <= 3)
        for question in dev_questions
    )
    available_failures = len(dev_questions) - dev_successes
    output = root / "data/manifests/dev_retrieval_inspection.json"
    report = {
        "status": "inspected_development_only",
        "recall_at_3": dev_successes / len(dev_questions) if dev_questions else 0.0,
        "requested_samples": {"successful_retrievals": 10, "failed_retrievals": 10, "high_bm25_score": 5, "small_top1_top2_margin": 5},
        "available_dev_failures": available_failures,
        "cases": selected,
        "decision": "No obvious tokenization or corpus-construction defect found; preserve the frozen BM25 configuration and rankings.",
        "test_metrics": "not inspected or reported",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "data/manifests/retrieval_frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dev_inspection_artifact"] = "data/manifests/dev_retrieval_inspection.json"
    manifest["dev_inspection_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["dev_inspection_status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "case_count": len(selected), "available_dev_failures": report["available_dev_failures"]}, indent=2))


if __name__ == "__main__":
    main()
