from scripts.tune_router import tune_threshold


def test_tunes_on_development_and_breaks_ties_with_less_retrieval():
    rows = [
        {"question_id": 1, "split": "dev", "condition": "no_rag", "exact_match": True},
        {"question_id": 1, "split": "dev", "condition": "imperfect_rag", "exact_match": False},
        {"question_id": 2, "split": "dev", "condition": "no_rag", "exact_match": False},
        {"question_id": 2, "split": "dev", "condition": "imperfect_rag", "exact_match": True},
        {"question_id": 3, "split": "test", "condition": "no_rag", "exact_match": False},
        {"question_id": 3, "split": "test", "condition": "imperfect_rag", "exact_match": True},
    ]
    result = tune_threshold(rows, {1: -1.0, 2: 1.0, 3: 0.0})

    assert result["selection_split"] == "dev"
    assert result["selected"]["em"] == 1.0
    assert result["selected"]["retrieval_count"] == 1
