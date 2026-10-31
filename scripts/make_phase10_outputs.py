"""Generate the requested Phase 10 tables and research figures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("no_rag", "imperfect_rag", "clean_rag")
GROUPS = ("overall", "head", "tail")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> float:
    return round(100 * value, 4)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs/phase10"
    figures = outputs / "figures"
    tables = outputs / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    questions = {row["question_id"]: row for row in read_jsonl(root / "data/frozen/popqa_sample.jsonl") if row.get("split") == "test"}
    scoring = read_jsonl(root / "outputs/test_scoring_ledger.jsonl")
    by_question: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scoring:
        by_question[row["question_id"]][row["condition"]] = row
    retrieval = read_jsonl(root / "data/retrieval/retrieval_results_top5.jsonl")
    top3: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval:
        if row["rank"] <= 3:
            top3[row["question_id"]].append(row)
    gold = {row["question_id"]: set(row["verified_gold_chunk_ids"]) for row in read_jsonl(root / "data/manifests/verified_gold_chunk_ids.jsonl")}
    retrieval_success = {
        question_id: any(row["is_verified_support"] for row in rows)
        for question_id, rows in top3.items()
    }

    dataset_rows = [
        {"relation": question["relation"], "group": question["popularity_group"], "count": sum(1 for q in questions.values() if q["relation"] == question["relation"] and q["popularity_group"] == question["popularity_group"])}
        for question in sorted(questions.values(), key=lambda row: (row["relation"], row["popularity_group"]))
    ]
    dataset_rows = list({(row["relation"], row["group"]): row for row in dataset_rows}.values())
    write_csv(tables / "dataset_composition.csv", dataset_rows)

    overall_rows = []
    head_tail_rows = []
    for condition in CONDITIONS:
        selected = [by_question[q][condition] for q in questions]
        overall_rows.append({"condition": condition, "n": len(selected), "em_percent": pct(sum(row["exact_match"] for row in selected) / len(selected)), "token_f1_percent": pct(sum(row["token_f1"] for row in selected) / len(selected))})
        for group in ("head", "tail"):
            ids = [q for q in questions if questions[q]["popularity_group"] == group]
            group_rows = [by_question[q][condition] for q in ids]
            head_tail_rows.append({"group": group, "condition": condition, "n": len(group_rows), "em_percent": pct(sum(row["exact_match"] for row in group_rows) / len(group_rows)), "token_f1_percent": pct(sum(row["token_f1"] for row in group_rows) / len(group_rows))})
    write_csv(tables / "overall_model_results.csv", overall_rows)
    write_csv(tables / "head_tail_results.csv", head_tail_rows)

    conditional_rows = []
    for success in (True, False):
        ids = [q for q in questions if retrieval_success.get(q, False) == success]
        for condition in ("no_rag", "imperfect_rag"):
            selected = [by_question[q][condition] for q in ids]
            conditional_rows.append({"retrieval_success": success, "condition": condition, "n": len(selected), "em_percent": pct(sum(row["exact_match"] for row in selected) / len(selected)), "token_f1_percent": pct(sum(row["token_f1"] for row in selected) / len(selected))})
    write_csv(tables / "conditional_retrieval_success.csv", conditional_rows)

    policy = json.loads((root / "outputs/policy_evaluation.json").read_text(encoding="utf-8"))
    policy_rows = [{"policy": name, "em_percent": pct(values["em"]), "token_f1_percent": pct(values["token_f1"]), "damage_rate_percent": pct(values["damage_rate"]), "rescue_rate_percent": pct(values["rescue_rate"]), "routed_to_rag_percent": pct(values["routed_to_rag"]), "median_latency_ms": values["latency_ms"]["median"], "p95_latency_ms": values["latency_ms"]["p95"]} for name, values in policy["policies"].items()]
    write_csv(tables / "policy_comparison.csv", policy_rows)

    paired = json.loads((root / "outputs/paired_statistics.json").read_text(encoding="utf-8"))
    paired_rows = []
    for group, values in paired["groups"].items():
        paired_rows.append({"group": group, "a_stable_correct": values["transition_table"]["a_stable_correct"], "b_damage": values["transition_table"]["b_damage"], "c_rescue": values["transition_table"]["c_rescue"], "d_stable_failure": values["transition_table"]["d_stable_failure"], "delta_em_percent_points": pct(values["bootstrap"]["delta_em"]["estimate"]), "delta_em_ci_low_percent_points": pct(values["bootstrap"]["delta_em"]["ci95"][0]), "delta_em_ci_high_percent_points": pct(values["bootstrap"]["delta_em"]["ci95"][1]), "damage_rate_ci_low_percent": pct(values["bootstrap"]["damage_rate"]["ci95"][0]), "damage_rate_ci_high_percent": pct(values["bootstrap"]["damage_rate"]["ci95"][1]), "rescue_rate_ci_low_percent": pct(values["bootstrap"]["rescue_rate"]["ci95"][0]), "rescue_rate_ci_high_percent": pct(values["bootstrap"]["rescue_rate"]["ci95"][1]), "mcnemar_p_value": values["mcnemar"]["p_value"]})
    write_csv(tables / "mcnemar_bootstrap_results.csv", paired_rows)

    errors = read_csv = list(csv.DictReader((root / "outputs/error_analysis.csv").open(encoding="utf-8")))
    error_counts = [{"error_category": category, "count": count} for category, count in sorted(Counter(row["error_category"] for row in errors).items(), key=lambda item: (-item[1], item[0]))]
    write_csv(tables / "error_category_counts.csv", error_counts)

    # Figure 1: EM by condition, split, and popularity group.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.24
    for index, condition in enumerate(CONDITIONS):
        values = [next(row["em_percent"] for row in head_tail_rows if row["group"] == group and row["condition"] == condition) if group != "overall" else next(row["em_percent"] for row in overall_rows if row["condition"] == condition) for group in GROUPS]
        ax.bar(x + (index - 1) * width, values, width, label=condition.replace("_", " ").title())
    ax.set_xticks(x, ["Overall", "Head", "Tail"]); ax.set_ylabel("Exact match (%)"); ax.set_ylim(0, 100); ax.set_title("Exact match by condition and popularity group"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "em_by_condition_group.png", dpi=180); plt.close(fig)

    # Figure 2: damage and rescue by popularity group.
    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(3); width = 0.35
    damage = [pct(paired["groups"][group]["bootstrap"]["damage_rate"]["estimate"]) for group in ("overall", "head", "tail")]
    rescue = [pct(paired["groups"][group]["bootstrap"]["rescue_rate"]["estimate"]) for group in ("overall", "head", "tail")]
    ax.bar(x - width / 2, damage, width, label="Damage"); ax.bar(x + width / 2, rescue, width, label="Rescue"); ax.set_xticks(x, ["Overall", "Head", "Tail"]); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 100); ax.set_title("Damage versus rescue"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "damage_vs_rescue.png", dpi=180); plt.close(fig)

    # Figure 3: Recall@3 by popularity group.
    recall_rows = []
    for group in ("head", "tail"):
        ids = [q for q in questions if questions[q]["popularity_group"] == group]
        hits = sum(bool(set(row["chunk_id"] for row in top3[q]) & gold.get(q, set())) for q in ids)
        recall_rows.append({"group": group, "n": len(ids), "recall_at_3_percent": pct(hits / len(ids))})
    write_csv(tables / "recall_at_3_by_group.csv", recall_rows)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.bar([row["group"].title() for row in recall_rows], [row["recall_at_3_percent"] for row in recall_rows]); ax.set_ylabel("Recall@3 (%)"); ax.set_ylim(0, 100); ax.set_title("Retrieval Recall@3 by popularity group"); fig.tight_layout(); fig.savefig(figures / "recall_at_3_by_group.png", dpi=180); plt.close(fig)

    # Figure 4: router trade-off.
    fig, ax = plt.subplots(figsize=(7, 5));
    for row in policy_rows:
        ax.scatter(row["routed_to_rag_percent"], row["em_percent"]); ax.annotate(row["policy"].replace("_", " "), (row["routed_to_rag_percent"], row["em_percent"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Routed to RAG (%)"); ax.set_ylabel("Exact match (%)"); ax.set_xlim(-5, 105); ax.set_ylim(0, 100); ax.set_title("Router trade-off: EM versus retrieval use"); fig.tight_layout(); fig.savefig(figures / "router_tradeoff.png", dpi=180); plt.close(fig)

    # Figure 5: policy latency.
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(policy_rows)); width = 0.36
    ax.bar(x - width / 2, [row["median_latency_ms"] for row in policy_rows], width, label="Median"); ax.bar(x + width / 2, [row["p95_latency_ms"] for row in policy_rows], width, label="P95"); ax.set_xticks(x, [row["policy"].replace("_", " ").title() for row in policy_rows], rotation=20, ha="right"); ax.set_ylabel("Latency (ms)"); ax.set_title("Policy latency"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "latency_median_p95.png", dpi=180); plt.close(fig)

    # Figure 6: optional transition heatmap.
    matrix = np.array([[paired["groups"]["overall"]["transition_table"][key] for key in ("a_stable_correct", "b_damage")] , [paired["groups"]["overall"]["transition_table"][key] for key in ("c_rescue", "d_stable_failure")]])
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix, cmap="Blues"); ax.set_xticks([0, 1], ["Imperfect correct", "Imperfect incorrect"]); ax.set_yticks([0, 1], ["No-RAG correct", "No-RAG incorrect"]); ax.set_title("Overall transition table");
    for (row, col), value in np.ndenumerate(matrix): ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Questions"); fig.tight_layout(); fig.savefig(figures / "transition_heatmap.png", dpi=180); plt.close(fig)

    source_paths = [root / "outputs/test_scoring_ledger.jsonl", root / "data/retrieval/retrieval_results_top5.jsonl", root / "outputs/policy_evaluation.json", root / "outputs/paired_statistics.json", root / "outputs/error_analysis.csv"]
    manifest = {"tables": sorted(str(path.relative_to(root)) for path in tables.iterdir()), "figures": sorted(str(path.relative_to(root)) for path in figures.iterdir()), "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}}
    (outputs / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tables": len(manifest["tables"]), "figures": len(manifest["figures"]), "manifest": str(outputs / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
