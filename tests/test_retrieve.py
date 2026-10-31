import json

from scripts.retrieve import retrieve_questions


def test_retrieve_saves_top_three_rows_and_router_features(tmp_path):
    chunks = [
        {"chunk_id": "c0", "page_title": "Page A", "text": "alpha beta"},
        {"chunk_id": "c1", "page_title": "Page B", "text": "alpha alpha gamma"},
        {"chunk_id": "c2", "page_title": "Page C", "text": "delta"},
        {"chunk_id": "c3", "page_title": "Page D", "text": "alpha"},
    ]
    tokenized = [{"chunk_id": row["chunk_id"], "tokens": row["text"].split()} for row in chunks]
    questions = [{"question_id": 7, "question": "alpha beta", "split": "dev"}]
    verified = {"7": {"c0"}}
    rows, features = retrieve_questions(chunks, tokenized, questions, verified_support=verified, k=3)

    assert len(rows) == 3
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert all({"question_id", "chunk_id", "page_title", "bm25_score", "chunk_text", "is_verified_support"} <= row.keys() for row in rows)
    assert rows[0]["is_verified_support"] is True
    assert features[0]["question_id"] == 7
    assert features[0]["s1"] >= features[0]["s2"]
    assert features[0]["score_margin"] == features[0]["s1"] - features[0]["s2"]
    assert features[0]["question_token_count"] == 2
    assert 0 <= features[0]["query_top_passage_token_overlap"] <= 1
