from scripts.standardize_bm25 import apply_standardizer, fit_dev_standardizer


def test_fit_uses_development_scores_only_and_applies_to_test():
    rows = [
        {"split": "dev", "s1": 1.0},
        {"split": "dev", "s1": 3.0},
        {"split": "test", "s1": 100.0},
    ]
    stats = fit_dev_standardizer(rows)

    assert stats == {"dev_count": 2, "dev_mean": 2.0, "dev_std": 1.0}
    assert apply_standardizer(rows, stats)[2]["s1_z"] == 98.0
