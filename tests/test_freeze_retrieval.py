from scripts.freeze_retrieval import development_metrics, freeze_gold_ids


def test_freezes_parent_chunk_ids_and_calculates_dev_metrics():
    review = [
        {
            "question_id": 1,
            "clean_doc_1": "[c0:sentence_01] support",
            "clean_doc_2": "[c1] context",
        },
        {"question_id": 2, "clean_doc_1": "[c3] support", "clean_doc_2": "[c4] context"},
    ]
    gold_rows = freeze_gold_ids(review)
    gold = {str(row["question_id"]): set(row["verified_gold_chunk_ids"]) for row in gold_rows}
    assert gold_rows[0]["related_context_chunk_ids"] == ["c1"]
    results = [
        {"question_id": 1, "rank": 1, "chunk_id": "wrong"},
        {"question_id": 1, "rank": 2, "chunk_id": "c0"},
        {"question_id": 1, "rank": 3, "chunk_id": "other"},
        {"question_id": 1, "rank": 4, "chunk_id": "c1"},
        {"question_id": 1, "rank": 5, "chunk_id": "last"},
        {"question_id": 2, "rank": 1, "chunk_id": "c3"},
        {"question_id": 2, "rank": 2, "chunk_id": "other"},
        {"question_id": 2, "rank": 3, "chunk_id": "other"},
        {"question_id": 2, "rank": 4, "chunk_id": "other"},
        {"question_id": 2, "rank": 5, "chunk_id": "other"},
    ]
    metrics = development_metrics(results, gold)
    assert metrics == {"questions": 2, "recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.75}
