import pytest

from scripts.step55_run_test import build_prediction_row, validate_resume_rows


def test_prediction_row_contains_required_test_set_fields():
    row = build_prediction_row(
        question_id=7,
        condition="clean_rag",
        prompt_hash="abc",
        input_token_count=12,
        output_token_count=2,
        raw_prediction="Jane Doe",
        latency_ms=3.5,
        retrieved_chunk_ids=["c1", "c2"],
        model_id="Qwen/example",
        model_revision="revision-1",
        generation_settings={"do_sample": False},
    )

    assert row == {
        "question_id": 7,
        "condition": "clean_rag",
        "prompt_hash": "abc",
        "input_token_count": 12,
        "output_token_count": 2,
        "raw_prediction": "Jane Doe",
        "latency_ms": 3.5,
        "retrieved_chunk_ids": ["c1", "c2"],
        "model_id": "Qwen/example",
        "model_revision": "revision-1",
        "generation_settings": {"do_sample": False},
    }


def test_resume_rejects_prompt_or_model_revision_mismatch():
    row = build_prediction_row(
        question_id=7,
        condition="no_rag",
        prompt_hash="wrong",
        input_token_count=1,
        output_token_count=1,
        raw_prediction="answer",
        latency_ms=1.0,
        retrieved_chunk_ids=[],
        model_id="model",
        model_revision="revision-1",
        generation_settings={},
    )

    with pytest.raises(ValueError, match="prompt hash"):
        validate_resume_rows([row], [7], "expected", "model", "revision-1")

    row["prompt_hash"] = "expected"
    row["model_revision"] = "revision-2"
    with pytest.raises(ValueError, match="model revision"):
        validate_resume_rows([row], [7], "expected", "model", "revision-1")
