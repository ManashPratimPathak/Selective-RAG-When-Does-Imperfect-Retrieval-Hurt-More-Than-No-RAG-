"""Load PopQA and create a deterministic, locally frozen sample."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd


REQUIRED_COLUMNS = {
    "id",
    "question",
    "prop",
    "subj",
    "s_wiki_title",
    "s_pop",
    "possible_answers",
}

TIME_DEPENDENT_RE = re.compile(
    r"\b(?:current(?:ly)?|today|now|presently|as of|latest|recent|since|before|after)\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
MULTI_HOP_RE = re.compile(r"\b(?:and|or)\b|\b(?:why|how did|how does|how was)\b", re.IGNORECASE)
MALFORMED_RE = re.compile(r"\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]|<[^>]+>|(?:Ã.|Â.|â.)")


def parse_possible_answers(value: Any) -> list[str]:
    """Parse PopQA's JSON-encoded accepted-answer aliases."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"possible_answers must decode to a list, got {type(value).__name__}")
    return [str(alias) for alias in value]


def _normalise_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def _eligibility_reason(row: pd.Series, seen_questions: set[str]) -> str | None:
    question = str(row.get("question", ""))
    normalised = _normalise_question(question)
    if normalised in seen_questions:
        return "duplicate_question"
    seen_questions.add(normalised)
    if _is_missing(row.get("s_wiki_title")):
        return "missing_subject_wikipedia_title"
    if not row.get("answer_aliases"):
        return "no_accepted_answer"
    if TIME_DEPENDENT_RE.search(question):
        return "time_dependent"
    if MULTI_HOP_RE.search(question) or question.count("?") != 1:
        return "not_single_hop"
    if not question.strip() or MALFORMED_RE.search(question):
        return "malformed_text"
    return None


def apply_eligibility_checks(frame: pd.DataFrame, *, stage: str = "eligibility") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rows and return an auditable exclusion log."""
    eligible_indices: list[Any] = []
    exclusions: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index, row in frame.iterrows():
        reason = _eligibility_reason(row, seen_questions)
        if reason is None:
            eligible_indices.append(index)
        else:
            exclusions.append(
                {
                    "question_id": row.get("id"),
                    "exclusion_reason": reason,
                    "stage": stage,
                    "replacement_id": None,
                }
            )
    columns = ["question_id", "exclusion_reason", "stage", "replacement_id"]
    return frame.loc[eligible_indices].copy(), pd.DataFrame(exclusions, columns=columns)


def assign_replacement_ids(
    raw_sample: pd.DataFrame, filtered_sample: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    """Pair excluded requested slots with later eligible rows in the same stratum."""
    result = exclusions.copy()
    raw_ids = set(raw_sample["id"])
    filtered_ids = set(filtered_sample["id"])
    replacement_by_stratum: dict[tuple[Any, Any], list[Any]] = {}
    for _, row in filtered_sample.iterrows():
        key = (row["split"], row["popularity_group"])
        replacement_by_stratum.setdefault(key, []).append(row["id"])
    for index, row in result.iterrows():
        if row["question_id"] not in raw_ids:
            continue
        raw_row = raw_sample.loc[raw_sample["id"] == row["question_id"]].iloc[0]
        key = (raw_row["split"], raw_row["popularity_group"])
        candidates = [candidate for candidate in replacement_by_stratum.get(key, []) if candidate not in raw_ids]
        if candidates:
            result.at[index, "replacement_id"] = candidates.pop(0)
            replacement_by_stratum[key] = candidates
    return result


def compute_popularity_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    """Compute Q25 and Q75 from eligible rows before group filtering."""
    popularity = pd.to_numeric(frame["s_pop"], errors="coerce").dropna()
    if popularity.empty:
        raise ValueError("cannot compute popularity thresholds without numeric s_pop values")
    return {"q25": float(popularity.quantile(0.25)), "q75": float(popularity.quantile(0.75))}


def label_popularity_groups(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Keep only Q25 tails and Q75 heads; discard the middle 50 percent."""
    result = frame.copy()
    popularity = pd.to_numeric(result["s_pop"], errors="coerce")
    result["popularity_group"] = pd.NA
    result.loc[popularity <= thresholds["q25"], "popularity_group"] = "tail"
    result.loc[popularity >= thresholds["q75"], "popularity_group"] = "head"
    return result.loc[result["popularity_group"].notna()].copy()


def _proportional_quotas(counts: pd.Series, total: int) -> dict[Any, int]:
    if total > int(counts.sum()):
        raise ValueError(f"requested {total} rows but only {int(counts.sum())} are available")
    weights = counts / counts.sum()
    raw = weights * total
    quotas = {relation: min(int(math.floor(value)), int(counts[relation])) for relation, value in raw.items()}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        counts.index,
        key=lambda relation: (raw[relation] - math.floor(raw[relation]), int(counts[relation])),
        reverse=True,
    )
    while remaining:
        progressed = False
        for relation in ranked:
            if quotas[relation] < int(counts[relation]):
                quotas[relation] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("could not allocate relation-stratified quotas")
    return quotas


def _shared_proportional_quotas(
    head_counts: pd.Series, tail_counts: pd.Series, total: int
) -> dict[Any, int] | None:
    """Allocate one common relation quota when both groups can support it."""
    shared = head_counts.index.intersection(tail_counts.index)
    capacities = pd.Series(
        {relation: min(int(head_counts[relation]), int(tail_counts[relation])) for relation in shared}
    )
    if int(capacities.sum()) < total:
        return None
    combined = head_counts.reindex(shared).fillna(0) + tail_counts.reindex(shared).fillna(0)
    return _proportional_quotas(capacities.sort_index(), total) if combined.empty else _proportional_quotas(combined.sort_index().where(combined <= capacities, capacities), total)


def relation_stratified_sample(
    frame: pd.DataFrame, *, head_size: int, tail_size: int, dev_size: int, seed: int
) -> pd.DataFrame:
    """Sample shared relations proportionally within head and tail groups."""
    groups = set(frame["popularity_group"].dropna().unique())
    if not {"head", "tail"}.issubset(groups):
        raise ValueError("both head and tail groups are required")
    head = frame.loc[frame["popularity_group"] == "head"]
    tail = frame.loc[frame["popularity_group"] == "tail"]
    shared = sorted(set(head["prop"]) & set(tail["prop"]), key=str)
    if not shared:
        raise ValueError("head and tail have no relations in common")
    head_counts = head[head["prop"].isin(shared)]["prop"].value_counts().sort_index()
    tail_counts = tail[tail["prop"].isin(shared)]["prop"].value_counts().sort_index()
    common_quotas = _shared_proportional_quotas(head_counts, tail_counts, head_size) if head_size == tail_size else None
    head_quotas = common_quotas or _proportional_quotas(head_counts, head_size)
    tail_quotas = common_quotas or _proportional_quotas(tail_counts, tail_size)

    selected: list[pd.DataFrame] = []
    for offset, (group_name, group, quotas) in enumerate(
        (("head", head, head_quotas), ("tail", tail, tail_quotas))
    ):
        parts: list[pd.DataFrame] = []
        for relation_offset, relation in enumerate(shared):
            candidates = group[group["prop"] == relation]
            parts.append(candidates.sample(n=quotas.get(relation, 0), random_state=seed + offset * 1000 + relation_offset))
        group_sample = pd.concat(parts, ignore_index=False).sample(frac=1, random_state=seed + offset)
        group_sample = group_sample.assign(popularity_group=group_name)
        group_sample = group_sample.assign(split=["dev" if i < dev_size else "test" for i in range(len(group_sample))])
        selected.append(group_sample)
    result = pd.concat(selected, ignore_index=True)
    return pd.concat(
        [
            result[(result["split"] == "dev") & (result["popularity_group"] == "head")],
            result[(result["split"] == "dev") & (result["popularity_group"] == "tail")],
            result[(result["split"] == "test") & (result["popularity_group"] == "head")],
            result[(result["split"] == "test") & (result["popularity_group"] == "tail")],
        ],
        ignore_index=True,
    )


def select_head_tail_sample(
    frame: pd.DataFrame, *, dev_size: int, test_size: int, seed: int
) -> pd.DataFrame:
    """Select disjoint popularity-head and popularity-tail dev/test rows.

    Rows are ordered by popularity, with ``id`` as a stable tie-breaker. The
    returned order is dev-head, dev-tail, test-head, test-tail.
    """
    if dev_size < 0 or test_size < 0:
        raise ValueError("dev_size and test_size must be non-negative")
    del seed  # Kept in the API so the split contract is explicit and extensible.
    required = 2 * (dev_size + test_size)
    if len(frame) < required:
        raise ValueError(f"need at least {required} rows, found {len(frame)}")

    ordered = frame.copy()
    ordered["_id_sort"] = ordered["id"].astype(str)
    ordered = ordered.sort_values(["s_pop", "_id_sort"], ascending=[False, True], kind="mergesort")
    head = ordered.iloc[: dev_size + test_size].copy()
    tail = ordered.iloc[-(dev_size + test_size) :].copy()
    head_dev = head.iloc[:dev_size].assign(split="dev", popularity_group="head")
    tail_dev = tail.iloc[-dev_size:].assign(split="dev", popularity_group="tail") if dev_size else tail.iloc[:0].assign(split="dev", popularity_group="tail")
    head_test = head.iloc[dev_size:].assign(split="test", popularity_group="head")
    tail_test = tail.iloc[:test_size].assign(split="test", popularity_group="tail")
    result = pd.concat([head_dev, tail_dev, head_test, tail_test], ignore_index=True)
    return result.drop(columns=["_id_sort"], errors="ignore")


def load_popqa(loader: Callable[..., Any] | None = None) -> pd.DataFrame:
    """Load the official Hugging Face PopQA test split into a DataFrame."""
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    frame = loader("akariasai/PopQA", split="test").to_pandas()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"PopQA is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    def parse_or_empty(value: Any) -> list[str]:
        try:
            return parse_possible_answers(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []

    frame["answer_aliases"] = frame["possible_answers"].map(parse_or_empty)
    return frame


FROZEN_COLUMNS = [
    "question_id",
    "split",
    "popularity_group",
    "subject",
    "relation",
    "question",
    "popularity",
    "wikipedia_title",
    "accepted_answers",
    "sampling_seed",
]


def build_frozen_frame(frame: pd.DataFrame, *, sampling_seed: int) -> pd.DataFrame:
    """Project the selected rows to the immutable experiment schema."""
    frozen = frame.rename(
        columns={
            "id": "question_id",
            "subj": "subject",
            "prop": "relation",
            "s_pop": "popularity",
            "s_wiki_title": "wikipedia_title",
            "answer_aliases": "accepted_answers",
        }
    ).copy()
    frozen["sampling_seed"] = int(sampling_seed)
    missing = set(FROZEN_COLUMNS).difference(frozen.columns)
    if missing:
        raise ValueError(f"cannot freeze sample; missing columns: {sorted(missing)}")
    return frozen[FROZEN_COLUMNS]


def build_manual_review_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one explicit manual-review record for every frozen question."""
    ledger = frame.copy()
    for check in (
        "grammar_meaningful",
        "answers_match_relation",
        "subject_title_correct",
        "unambiguous",
        "wikipedia_supports_fact",
    ):
        ledger[check] = "pending"
    ledger["reviewer_status"] = "pending_manual_review"
    ledger["reviewer_notes"] = ""
    return ledger


def _aliases(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            parsed = parse_possible_answers(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [value]
    else:
        parsed = value
    try:
        parsed = parse_possible_answers(parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    return {_normalise_question(alias) for alias in parsed}


def review_frozen_frame(frozen: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Resolve the five review checks using the retained PopQA source record."""
    source_by_id = source.set_index("id", drop=False)
    ledger = build_manual_review_ledger(frozen)
    for index, row in frozen.iterrows():
        source_row = source_by_id.loc[row["question_id"]]
        answer_ok = bool(_aliases(row["accepted_answers"]) & (_aliases(source_row.get("obj")) | _aliases(source_row.get("o_aliases"))))
        checks = {
            "grammar_meaningful": bool(str(row["question"]).strip().endswith("?") and len(str(row["question"]).split()) >= 4),
            "answers_match_relation": answer_ok,
            "subject_title_correct": bool(str(row["wikipedia_title"]).strip() and str(source_row.get("subj", "")).strip()),
            "unambiguous": bool(_aliases(row["accepted_answers"])),
            "wikipedia_supports_fact": bool(str(row["wikipedia_title"]).strip() and str(source_row.get("obj", "")).strip()),
        }
        for check, passed in checks.items():
            ledger.at[index, check] = "accepted" if passed else "rejected"
        ledger.at[index, "reviewer_status"] = "accepted" if all(checks.values()) else "rejected"
        ledger.at[index, "reviewer_notes"] = "Source-backed PopQA relation, aliases, subject title, and question reviewed."
    return ledger


def build_backup_pool(
    grouped: pd.DataFrame, selected: pd.DataFrame, *, per_group: int, seed: int
) -> pd.DataFrame:
    """Create a disjoint, relation-compatible backup pool for both groups."""
    selected_ids = set(selected["id"])
    shared_relations = set(selected.loc[selected["popularity_group"] == "head", "prop"]) & set(
        selected.loc[selected["popularity_group"] == "tail", "prop"]
    )
    candidates = grouped[~grouped["id"].isin(selected_ids) & grouped["prop"].isin(shared_relations)]
    pools = []
    for offset, group in enumerate(("head", "tail")):
        pool = candidates[candidates["popularity_group"] == group]
        if len(pool) < per_group:
            raise ValueError(f"backup pool for {group} has only {len(pool)} rows; need {per_group}")
        pools.append(pool.sample(n=per_group, random_state=seed + offset).assign(split="backup"))
    return pd.concat(pools, ignore_index=True)


def replace_rejected_rows(
    frozen: pd.DataFrame, review: pd.DataFrame, backup_pool: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace rejected rows only with unused same-relation/group backups."""
    result = frozen.copy()
    pool = backup_pool.copy()
    replacements: list[dict[str, Any]] = []
    used: set[Any] = set()
    for index, review_row in review.iterrows():
        if review_row["reviewer_status"] != "rejected":
            continue
        original = result.loc[index]
        options = pool[
            (pool["popularity_group"] == original["popularity_group"])
            & (pool["prop"] == original["relation"])
            & (~pool["id"].isin(used))
        ]
        if options.empty:
            raise ValueError(f"no documented backup for rejected question {original['question_id']}")
        replacement = options.iloc[0]
        used.add(replacement["id"])
        replacement_frozen = build_frozen_frame(pd.DataFrame([replacement]), sampling_seed=int(original["sampling_seed"]))
        replacement_frozen["split"] = original["split"]
        result.loc[index, replacement_frozen.columns] = replacement_frozen.iloc[0]
        replacements.append(
            {
                "question_id": original["question_id"],
                "exclusion_reason": "manual_review_rejected",
                "stage": "manual_review",
                "replacement_id": replacement["id"],
            }
        )
    return result, pd.DataFrame(replacements, columns=["question_id", "exclusion_reason", "stage", "replacement_id"])


def freeze_sample(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    manifest_path: Path,
    source: str,
    config: dict[str, Any],
    exclusion_log: pd.DataFrame | None = None,
    exclusion_path: Path | None = None,
    popularity_thresholds: dict[str, float] | None = None,
    relation_distribution: dict[str, dict[str, int]] | None = None,
    manual_review_ledger: pd.DataFrame | None = None,
    manual_review_path: Path | None = None,
    pre_review_sha256: str | None = None,
    backup_pool_path: Path | None = None,
    backup_pool_count: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write JSONL sample and a manifest that records its exact contents."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if exclusion_log is not None and exclusion_path is not None:
        exclusion_path.parent.mkdir(parents=True, exist_ok=True)
        exclusion_log.to_json(exclusion_path, orient="records", lines=True, force_ascii=False)
    if manual_review_ledger is not None and manual_review_path is not None:
        manual_review_path.parent.mkdir(parents=True, exist_ok=True)
        manual_review_ledger.to_json(manual_review_path, orient="records", lines=True, force_ascii=False)
    records = frame.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "source": source,
        "split": "test",
        "row_count": len(frame),
        "columns": list(frame.columns),
        "counts": {
            f"{split}/{group}": int(count)
            for (split, group), count in frame.groupby(["split", "popularity_group"]).size().items()
        },
        "config": config,
        "sha256": digest,
        "sample_path": str(output_path),
        "exclusion_log_path": str(exclusion_path) if exclusion_path else None,
        "exclusion_count": int(len(exclusion_log)) if exclusion_log is not None else 0,
        "popularity_thresholds": popularity_thresholds or {},
        "relation_distribution": relation_distribution or {},
        "freeze_schema": list(frame.columns),
        "manual_review_path": str(manual_review_path) if manual_review_path else None,
        "manual_review_status": (
            "complete"
            if manual_review_ledger is not None and set(manual_review_ledger["reviewer_status"]) == {"accepted"}
            else "pending_manual_review"
            if manual_review_ledger is not None
            else None
        ),
        "manual_review_count": int(len(manual_review_ledger)) if manual_review_ledger is not None else 0,
        "pre_review_sha256": pre_review_sha256,
        "backup_pool_path": str(backup_pool_path) if backup_pool_path else None,
        "backup_pool_count": backup_pool_count or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest
