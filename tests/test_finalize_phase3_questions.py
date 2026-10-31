from scripts.finalize_phase3_questions import clarify_question, sanitize_aliases, reconcile_review_row, clarification_ids


def test_clarify_question_preserves_original_and_disambiguates_subject():
    row = {
        "question_id": 1,
        "question": "Who produced Don?",
        "subject": "Don",
        "wikipedia_title": "Don (2006 Hindi film)",
    }

    result = clarify_question(row)

    assert result["source_question"] == "Who produced Don?"
    assert result["question"] == "Who produced Don (2006 Hindi film)?"
    assert row["question"] == "Who produced Don?"


def test_clarify_question_leaves_exact_title_unchanged():
    row = {"question": "What genre is Kagemusha?", "subject": "Kagemusha", "wikipedia_title": "Kagemusha"}

    assert clarify_question(row) == row


def test_clarify_question_matches_subject_as_whole_word():
    row = {"question_id": 4315195, "question": "Who was the producer of H?", "subject": "H", "wikipedia_title": "H (1990 film)"}

    assert clarify_question(row)["question"] == "Who was the producer of H (1990 film)?"


def test_sanitize_aliases_removes_rights_identifier_but_preserves_names():
    assert sanitize_aliases(["John Cale", "00033982178 IPI", "John Davies Cale"]) == [
        "John Cale", "John Davies Cale"
    ]


def test_clarification_resolves_adoptive_parent_and_series_director():
    father = {"question_id": 4145108, "question": "Who is the father of Trace Cyrus?", "subject": "Trace Cyrus", "wikipedia_title": "Trace Cyrus"}
    series = {"question_id": 930027, "question": "Who was the director of The Missing?", "subject": "The Missing", "wikipedia_title": "The Missing (British TV series)"}

    assert clarify_question(father)["question"] == "Who is the adoptive father of Trace Cyrus?"
    assert clarify_question(series)["question"] == "Who directed the first series of The Missing (British TV series)?"


def test_review_remains_accepted_when_the_frozen_question_is_unchanged():
    question = {"question_id": 1, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"], "wikipedia_title": "Example"}
    review = {**question, "reviewer_status": "accepted", "grammar_meaningful": "accepted", "reviewer_notes": "Reviewed."}

    assert reconcile_review_row(question, review) == review


def test_changed_question_returns_to_pending_review():
    question = {"question_id": 1, "question": "Who directed Example (film)?", "accepted_answers": ["Jane Doe"], "wikipedia_title": "Example (film)"}
    review = {**question, "question": "Who directed Example?", "reviewer_status": "accepted", "grammar_meaningful": "accepted"}

    updated = reconcile_review_row(question, review)

    assert updated["reviewer_status"] == "pending_ai_review"
    assert updated["grammar_meaningful"] == "pending"


def test_clarification_history_comes_from_frozen_source_question():
    rows = [
        {"question_id": 1, "source_question": "Who directed Example?", "question": "Who directed Example (film)?"},
        {"question_id": 2, "question": "Who directed Another?"},
    ]

    assert clarification_ids(rows) == [1]
