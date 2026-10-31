from scripts.reconcile_question_review import reconcile_review_rows


def test_reconcile_marks_missing_question_reviews_pending_and_drops_stale_ids():
    questions = [
        {"question_id": 1, "question": "Who wrote A?"},
        {"question_id": 2, "question": "Who wrote B?"},
    ]
    reviews = [
        {"question_id": 1, "question": "Old wording", "reviewer_status": "accepted"},
        {"question_id": 3, "reviewer_status": "accepted"},
    ]

    updated = reconcile_review_rows(questions, reviews)

    assert [row["question_id"] for row in updated] == [1, 2]
    assert updated[0]["question"] == "Who wrote A?"
    assert updated[0]["reviewer_status"] == "accepted"
    assert updated[1]["reviewer_status"] == "pending_ai_review"
    assert updated[1]["wikipedia_supports_fact"] == "pending"
