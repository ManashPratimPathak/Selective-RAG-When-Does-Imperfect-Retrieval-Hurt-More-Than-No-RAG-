from scripts.step56_measure_latency import nearest_rank_percentile, summarize_latency


def test_nearest_rank_percentile_is_deterministic():
    assert nearest_rank_percentile([1, 2, 3, 4], 50) == 2
    assert nearest_rank_percentile([1, 2, 3, 4], 95) == 4


def test_latency_summary_is_split_by_condition():
    rows = [
        {"condition": "no_rag", "generation_latency_ms": 2.0},
        {"condition": "no_rag", "generation_latency_ms": 4.0},
        {"condition": "clean_rag", "generation_latency_ms": 8.0},
        {"condition": "clean_rag", "generation_latency_ms": 10.0},
        {"condition": "imperfect_rag", "generation_latency_ms": 20.0},
        {"condition": "imperfect_rag", "generation_latency_ms": 30.0},
    ]
    summary = summarize_latency(rows, ("generation_latency_ms",))
    assert set(summary) == {"no_rag", "clean_rag", "imperfect_rag"}
    assert summary["clean_rag"]["generation_latency_ms"]["median"] == 9.0
