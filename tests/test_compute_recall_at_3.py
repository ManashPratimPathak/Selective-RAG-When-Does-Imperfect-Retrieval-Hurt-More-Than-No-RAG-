from scripts.compute_recall_at_3 import compute


def test_recall_counts_only_verified_support_ids():
    result = compute(
        [
            {"question_id": 1, "rank": 1, "chunk_id": "context"},
            {"question_id": 1, "rank": 2, "chunk_id": "support"},
            {"question_id": 1, "rank": 3, "chunk_id": "other"},
            {"question_id": 2, "rank": 1, "chunk_id": "same-page"},
            {"question_id": 2, "rank": 2, "chunk_id": "other"},
            {"question_id": 2, "rank": 3, "chunk_id": "other2"},
        ],
        [{"question_id": 1, "split": "dev"}, {"question_id": 2, "split": "dev"}],
        [
            {"question_id": 1, "verified_gold_chunk_ids": ["support"], "related_context_chunk_ids": ["context"]},
            {"question_id": 2, "verified_gold_chunk_ids": ["support2"], "related_context_chunk_ids": ["same-page"]},
        ],
    )
    assert result["hits_at_3"] == 1
    assert result["misses_at_3"] == 1
    assert result["recall_at_3"] == 0.5
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_5"] == 0.5
    assert result["mrr"] == 0.25
