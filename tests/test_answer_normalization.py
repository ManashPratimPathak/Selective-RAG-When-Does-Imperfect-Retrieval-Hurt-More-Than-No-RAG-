import importlib


scoring = importlib.import_module("scripts.08_score")
score_prediction = scoring.score_prediction


def test_score_normalizes_prediction_and_aliases_but_keeps_raw_values():
    result = score_prediction("  The Eiffel-Tower!  ", ["eiffel tower", "An answer"])

    assert result == {
        "raw_prediction": "  The Eiffel-Tower!  ",
        "normalized_prediction": "eiffeltower",
        "raw_aliases": ["eiffel tower", "An answer"],
        "normalized_aliases": ["eiffel tower", "answer"],
        "exact_match": False,
        "token_f1": 0.0,
    }


def test_articles_are_removed_as_tokens_only():
    assert scoring.normalize_answer("the theatre and another") == "theatre and another"


def test_alias_aware_match_requires_full_equality():
    assert scoring.score_prediction("York", ["New York"])["exact_match"] is False
    assert scoring.score_prediction("New York", ["New York"])["exact_match"] is True


def test_empty_prediction_cannot_match_empty_alias():
    result = scoring.score_prediction("!!!", ["---"])
    assert result["exact_match"] is False
    assert result["token_f1"] == 0.0


def test_token_f1_uses_maximum_alias_score():
    result = scoring.score_prediction("New York City", ["New York", "Los Angeles"])
    assert result["token_f1"] == 0.8


def test_damage_rate_uses_only_no_rag_correct_questions():
    rows = [
        {"question_id": 1, "condition": "no_rag", "exact_match": True},
        {"question_id": 1, "condition": "imperfect_rag", "exact_match": False},
        {"question_id": 2, "condition": "no_rag", "exact_match": True},
        {"question_id": 2, "condition": "imperfect_rag", "exact_match": True},
        {"question_id": 3, "condition": "no_rag", "exact_match": False},
        {"question_id": 3, "condition": "imperfect_rag", "exact_match": False},
    ]
    assert scoring.damage_rate(rows) == {"damaged": 1, "baseline_correct": 2, "rate": 0.5}


def test_rescue_rate_uses_only_no_rag_incorrect_questions():
    rows = [
        {"question_id": 1, "condition": "no_rag", "exact_match": False},
        {"question_id": 1, "condition": "imperfect_rag", "exact_match": True},
        {"question_id": 2, "condition": "no_rag", "exact_match": False},
        {"question_id": 2, "condition": "imperfect_rag", "exact_match": False},
        {"question_id": 3, "condition": "no_rag", "exact_match": True},
        {"question_id": 3, "condition": "imperfect_rag", "exact_match": True},
    ]
    assert scoring.rescue_rate(rows) == {"rescued": 1, "baseline_incorrect": 2, "rate": 0.5}


def test_accuracy_change_is_reported_in_percentage_points():
    accuracy = {
        "no_rag": {"accuracy": 0.13},
        "imperfect_rag": {"accuracy": 0.615},
    }
    assert scoring.accuracy_change(accuracy) == {
        "imperfect_rag_minus_no_rag": 0.485,
        "percentage_points": 48.5,
    }


def test_transition_table_and_paired_statistics():
    rows = [
        {"question_id": 1, "condition": "no_rag", "exact_match": True},
        {"question_id": 1, "condition": "imperfect_rag", "exact_match": True},
        {"question_id": 2, "condition": "no_rag", "exact_match": True},
        {"question_id": 2, "condition": "imperfect_rag", "exact_match": False},
        {"question_id": 3, "condition": "no_rag", "exact_match": False},
        {"question_id": 3, "condition": "imperfect_rag", "exact_match": True},
        {"question_id": 4, "condition": "no_rag", "exact_match": False},
        {"question_id": 4, "condition": "imperfect_rag", "exact_match": False},
    ]
    assert scoring.transition_table(rows) == {
        "stable_correct": 1,
        "damage": 1,
        "rescue": 1,
        "stable_failure": 1,
    }
    stats = scoring.paired_statistics(rows, samples=100, seed=0)
    assert stats["mcnemar_damage"] == 1
    assert stats["mcnemar_rescue"] == 1
    assert stats["mcnemar_p_value"] == 1.0
