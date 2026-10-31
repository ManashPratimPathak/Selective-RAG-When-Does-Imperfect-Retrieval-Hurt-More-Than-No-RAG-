"""Compute the canonical development Recall@3 from frozen gold chunk IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute(results: list[dict[str, Any]], questions: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = {str(row["question_id"]): set(row["verified_gold_chunk_ids"]) for row in gold_rows}
    dev = [row for row in questions if row.get("split") == "dev"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row["question_id"]), []).append(row)
    hit_ids: list[int] = []
    miss_ids: list[int] = []
    decisions = []
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []
    for question in dev:
        question_id = str(question["question_id"])
        ranked = sorted(grouped[question_id], key=lambda row: row["rank"])
        matched = [row["chunk_id"] for row in ranked[:3] if row["chunk_id"] in gold[question_id]]
        hit = bool(matched)
        (hit_ids if hit else miss_ids).append(question["question_id"])
        first_rank = next((row["rank"] for row in ranked if row["chunk_id"] in gold[question_id]), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in hits:
            hits[k] += int(any(row["chunk_id"] in gold[question_id] for row in ranked[:k]))
        decisions.append({"question_id": question["question_id"], "hit": hit, "matched_gold_chunk_ids": matched, "first_relevant_rank": first_rank, "top3_chunk_ids": [row["chunk_id"] for row in ranked[:3]]})
    return {
        "status": "canonical_frozen",
        "split": "dev",
        "questions": len(dev),
        "hits_at_3": len(hit_ids),
        "misses_at_3": len(miss_ids),
        "recall_at_3": len(hit_ids) / len(dev) if dev else 0.0,
        "hits_at_1": hits[1],
        "recall_at_1": hits[1] / len(dev) if dev else 0.0,
        "hits_at_5": hits[5],
        "recall_at_5": hits[5] / len(dev) if dev else 0.0,
        "mrr": sum(reciprocal_ranks) / len(dev) if dev else 0.0,
        "hit_question_ids": hit_ids,
        "miss_question_ids": miss_ids,
        "decisions": decisions,
        "gold_rule": "A hit requires a top-three chunk_id in that question's verified_gold_chunk_ids; aliases, same-page matches, related context chunks, and unverified chunks do not count.",
        "supersedes": {"previous_recall_at_3": 0.85, "previous_hits_at_3": 34, "previous_questions": 40, "reason": "The earlier evaluator counted related clean_doc_2 context chunks. The canonical evaluator counts only frozen relational-support gold chunk IDs."},
        "test_metrics": "not computed or reported",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data/manifests/recall_at_3_canonical.json"
    result = compute(
        _read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl"),
        _read_jsonl(root / "data/frozen/popqa_sample.jsonl"),
        _read_jsonl(root / "data/manifests/verified_gold_chunk_ids.jsonl"),
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "data/manifests/retrieval_frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_recall_at_3_artifact"] = "data/manifests/recall_at_3_canonical.json"
    manifest["canonical_recall_at_3_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["development_metrics"] = {
        "questions": result["questions"],
        "recall_at_1": result["recall_at_1"],
        "recall_at_3": result["recall_at_3"],
        "recall_at_5": result["recall_at_5"],
        "mrr": result["mrr"],
    }
    manifest["superseded_recall_at_3"] = {"recall_at_3": 0.85, "hits_at_3": 34, "reason": "Related context chunks were incorrectly counted as relational-support gold."}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "hits_at_3", "misses_at_3")}, indent=2))


if __name__ == "__main__":
    main()
