import json
from pathlib import Path

import pandas as pd

from scripts.prepare_popqa import (
    apply_eligibility_checks,
    build_manual_review_ledger,
    review_frozen_frame,
    build_frozen_frame,
    compute_popularity_thresholds,
    freeze_sample,
    label_popularity_groups,
    parse_possible_answers,
    relation_stratified_sample,
    select_head_tail_sample,
)


def test_parse_possible_answers_uses_json_and_preserves_aliases():
    raw = json.dumps(["New York", "NYC"])

    assert parse_possible_answers(raw) == ["New York", "NYC"]


def test_select_head_tail_sample_is_disjoint_and_deterministic():
    frame = pd.DataFrame(
        {
            "id": list(range(8)),
            "s_pop": [80, 70, 60, 50, 40, 30, 20, 10],
        }
    )

    first = select_head_tail_sample(frame, dev_size=2, test_size=2, seed=42)
    second = select_head_tail_sample(frame, dev_size=2, test_size=2, seed=42)

    assert first.equals(second)
    assert first["split"].tolist() == ["dev", "dev", "dev", "dev", "test", "test", "test", "test"]
    assert set(first.loc[first["split"] == "dev", "id"]) == {0, 1, 6, 7}
    assert set(first.loc[first["split"] == "test", "id"]) == {2, 3, 4, 5}


def test_freeze_sample_writes_json_serializable_manifest(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "split": ["dev", "test"],
            "popularity_group": ["head", "tail"],
            "answer_aliases": [["A"], ["B"]],
        }
    )

    freeze_sample(
        frame,
        output_path=tmp_path / "sample.jsonl",
        manifest_path=tmp_path / "manifest.json",
        source="test",
        config={"seed": 42},
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"dev/head": 1, "test/tail": 1}


def test_apply_eligibility_checks_logs_each_exclusion_reason():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "question": [
                "What is the occupation of Ada Lovelace?",
                "What is the occupation of Ada Lovelace?",
                "Who is the current president of France?",
                "What is the occupation of Ada and Alan?",
                "What is the occupation of Ada Lovelace? \ufffd",
                "What is the occupation of Grace Hopper?",
            ],
            "s_wiki_title": ["Ada Lovelace", "Ada Lovelace", "France", "Ada Lovelace", "Ada Lovelace", "Grace Hopper"],
            "answer_aliases": [["writer"], ["writer"], ["president"], ["writer"], ["writer"], []],
        }
    )

    eligible, exclusions = apply_eligibility_checks(frame)

    assert eligible["id"].tolist() == [1]
    assert exclusions[["question_id", "exclusion_reason", "replacement_id"]].to_dict("records") == [
        {"question_id": 2, "exclusion_reason": "duplicate_question", "replacement_id": None},
        {"question_id": 3, "exclusion_reason": "time_dependent", "replacement_id": None},
        {"question_id": 4, "exclusion_reason": "not_single_hop", "replacement_id": None},
        {"question_id": 5, "exclusion_reason": "malformed_text", "replacement_id": None},
        {"question_id": 6, "exclusion_reason": "no_accepted_answer", "replacement_id": None},
    ]


def test_popularity_groups_use_quartiles_and_drop_the_middle():
    frame = pd.DataFrame({"id": range(8), "s_pop": range(8), "prop": ["p"] * 8})

    thresholds = compute_popularity_thresholds(frame)
    grouped = label_popularity_groups(frame, thresholds)

    assert thresholds == {"q25": 1.75, "q75": 5.25}
    assert grouped["id"].tolist() == [0, 1, 6, 7]
    assert grouped["popularity_group"].tolist() == ["tail", "tail", "head", "head"]


def test_relation_stratified_sample_uses_only_shared_relations_and_seed():
    frame = pd.DataFrame(
        {
            "id": range(12),
            "prop": ["occupation"] * 3 + ["birthplace"] * 3 + ["occupation"] * 2 + ["birthplace"] * 2 + ["unshared"] * 2,
            "popularity_group": ["head"] * 6 + ["tail"] * 6,
        }
    )

    first = relation_stratified_sample(frame, head_size=3, tail_size=3, dev_size=1, seed=42)
    second = relation_stratified_sample(frame, head_size=3, tail_size=3, dev_size=1, seed=42)

    assert first.equals(second)
    assert set(first["prop"]) == {"occupation", "birthplace"}
    assert first.groupby("popularity_group").size().to_dict() == {"head": 3, "tail": 3}


def test_frozen_frame_has_only_protocol_fields_and_sampling_seed():
    frame = pd.DataFrame(
        {
            "id": [7],
            "split": ["test"],
            "popularity_group": ["head"],
            "subj": ["Ada Lovelace"],
            "prop": ["occupation"],
            "question": ["What was Ada Lovelace's occupation?"],
            "s_pop": [99],
            "s_wiki_title": ["Ada Lovelace"],
            "answer_aliases": [["writer"]],
            "unused": ["not frozen"],
        }
    )

    frozen = build_frozen_frame(frame, sampling_seed=42)
    review = build_manual_review_ledger(frozen)

    assert list(frozen.columns) == [
        "question_id", "split", "popularity_group", "subject", "relation",
        "question", "popularity", "wikipedia_title", "accepted_answers", "sampling_seed",
    ]
    assert frozen.iloc[0]["sampling_seed"] == 42
    assert review.iloc[0]["reviewer_status"] == "pending_manual_review"


def test_review_frozen_frame_resolves_all_source-backed_checks():
    frozen = pd.DataFrame(
        {
            "question_id": [7], "split": ["test"], "popularity_group": ["head"],
            "subject": ["Ada"], "relation": ["occupation"], "question": ["What is Ada occupation?"],
            "popularity": [99], "wikipedia_title": ["Ada"], "accepted_answers": [["writer"]],
            "sampling_seed": [42],
        }
    )
    source = pd.DataFrame({"id": [7], "subj": ["Ada"], "prop": ["occupation"], "obj": ["writer"], "o_aliases": [["writer"]]})

    reviewed = review_frozen_frame(frozen, source)

    assert reviewed.iloc[0]["reviewer_status"] == "accepted"
    assert all(reviewed.iloc[0][column] == "accepted" for column in [
        "grammar_meaningful", "answers_match_relation", "subject_title_correct", "unambiguous", "wikipedia_supports_fact"
    ])
