"""Deterministic sampling of natural PopQA/Wikipedia distractor pages.

Distractors are page titles already present in PopQA.  This module never
creates text or facts; it only records which existing Wikipedia pages should
be fetched and why they were eligible.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


TAIL_THRESHOLD = 227.0
HEAD_THRESHOLD = 5395.5


def _title(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def _key(value: Any) -> str:
    return _title(value).casefold()


def excluded_subject_titles(selected: pd.DataFrame, backup: pd.DataFrame, source: pd.DataFrame) -> set[str]:
    """Return subject-page titles reserved by sample or backup rows."""
    titles: set[str] = set()
    for frame, column in ((selected, "wikipedia_title"), (backup, "wikipedia_title")):
        if column in frame:
            titles.update(_key(value) for value in frame[column] if _title(value))
    selected_ids = set(selected["question_id"].tolist()) if "question_id" in selected else set()
    if "id" in source and selected_ids and "s_wiki_title" in source:
        selected_source = source[source["id"].isin(selected_ids)]
        titles.update(_key(value) for value in selected_source["s_wiki_title"] if _title(value))
    return titles


def build_distractor_candidates(
    source: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    backup: pd.DataFrame | None = None,
    relations: set[str] | None = None,
    tail_threshold: float = TAIL_THRESHOLD,
    head_threshold: float = HEAD_THRESHOLD,
) -> pd.DataFrame:
    """Build subject-page candidates classified by subject popularity."""
    required = {"id", "prop", "s_wiki_title", "s_pop"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"PopQA source is missing columns: {sorted(missing)}")

    selected_ids = set(selected["question_id"].tolist()) if "question_id" in selected else set()
    excluded_titles = excluded_subject_titles(selected, backup if backup is not None else pd.DataFrame(), source)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        if row["id"] in selected_ids:
            continue
        relation = str(row["prop"])
        if relations is not None and relation not in relations:
            continue
        page_title = _title(row["s_wiki_title"])
        popularity = float(row["s_pop"])
        group = "tail" if popularity <= tail_threshold else "head" if popularity >= head_threshold else None
        if not page_title or group is None or _key(page_title) in excluded_titles:
            continue
        rows.append(
            {
                "page_title": page_title,
                "entity_group": group,
                "subject_popularity": popularity,
                "relation": relation,
                "source_question_id": int(row["id"]),
                "source_entity": _title(row["subj"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["page_title", "entity_group", "subject_popularity", "relation", "source_question_id", "source_entity"])
    candidates = pd.DataFrame(rows)
    # Keep the first deterministic provenance record for each page/group.
    candidates = (
        candidates.sort_values(["entity_group", "page_title", "relation", "source_question_id"], kind="mergesort")
        .groupby(["entity_group", "page_title"], sort=True, as_index=False)
        .agg(
            subject_popularity=("subject_popularity", "first"),
            relation=("relation", lambda values: ",".join(sorted(set(values)))),
            source_question_id=("source_question_id", "min"),
            source_entity=("source_entity", "first"),
        )
    )
    return candidates


def sample_distractor_pages(
    candidates: pd.DataFrame,
    *,
    count: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample exactly ``count`` unique subject pages from one group."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if candidates.empty or count == 0:
        return candidates.iloc[:0].copy()
    work = candidates.drop_duplicates("page_title").sort_values("page_title", kind="mergesort")
    if len(work) < count:
        raise ValueError(f"need {count} candidates, found {len(work)}")
    return work.sample(n=count, random_state=seed).sort_values("page_title", kind="mergesort").reset_index(drop=True)


def selected_relations(sample: pd.DataFrame) -> set[str]:
    """Return the relations represented by the frozen experiment sample."""
    return {str(value) for value in sample["relation"].dropna()}


def choose_distractor_target(
    dev_recall_at_3: float | None,
    *,
    current_target: int = 500,
    expanded_target: int = 1000,
    extremely_high_threshold: float = 0.95,
) -> int:
    """Choose 500 vs 1,000 using development retrieval only.

    ``None`` means the development gate has not run yet, so the conservative
    initial target is returned.  Test metrics are intentionally not accepted
    by this function or the CLI.
    """
    if not 0 <= extremely_high_threshold <= 1:
        raise ValueError("extremely_high_threshold must be between 0 and 1")
    if dev_recall_at_3 is None:
        return current_target
    if not 0 <= dev_recall_at_3 <= 1:
        raise ValueError("dev_recall_at_3 must be between 0 and 1")
    return expanded_target if dev_recall_at_3 >= extremely_high_threshold else current_target
