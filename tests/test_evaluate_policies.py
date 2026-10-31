from scripts.evaluate_policies import evaluate_policy


def test_policy_metrics_use_selected_answers_and_baseline_pairs():
    rows = [
        {"question_id": 1, "condition": "no_rag", "exact_match": True, "token_f1": 1.0},
        {"question_id": 1, "condition": "imperfect_rag", "exact_match": False, "token_f1": 0.0},
        {"question_id": 1, "condition": "clean_rag", "exact_match": True, "token_f1": 1.0},
        {"question_id": 2, "condition": "no_rag", "exact_match": False, "token_f1": 0.0},
        {"question_id": 2, "condition": "imperfect_rag", "exact_match": True, "token_f1": 0.5},
        {"question_id": 2, "condition": "clean_rag", "exact_match": True, "token_f1": 1.0},
    ]
    result = evaluate_policy("threshold_router", rows, {(1, "no_rag"): 10.0, (2, "imperfect_rag"): 20.0}, {1: "no_rag", 2: "imperfect_rag"})

    assert result["em"] == 1.0
    assert result["token_f1"] == 0.75
    assert result["damage_rate"] == 0.0
    assert result["rescue_rate"] == 1.0
    assert result["routed_to_rag"] == 0.5
    assert result["latency_ms"] == {"median": 15.0, "p95": 20.0}
