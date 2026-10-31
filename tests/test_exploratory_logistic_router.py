from scripts.exploratory_logistic_router import run_exploratory_router


def test_exploratory_router_uses_requested_pipeline_features():
    x = [[-1.0, 1.0, 0.1, 5.0], [1.0, -1.0, 0.9, 10.0]] * 5
    y = [0, 1] * 5
    result = run_exploratory_router(x, y)

    assert result["feature_names"] == ["s1", "score_margin", "query_top_passage_token_overlap", "question_token_count"]
    assert result["cross_validation"] == {"method": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 0}
    assert result["five_fold_scores"]
    assert "exploratory" in result["note"].lower()
