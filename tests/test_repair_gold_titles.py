from scripts.repair_gold_titles import apply_title_corrections


def test_title_correction_replaces_shell_and_preserves_question_identity():
    questions = [{"question_id": 7, "wikipedia_title": "Old title", "question": "Who directed Film?"}]
    pages = [{"requested_title": "Old title", "page_id": 10, "raw_html": "shell"}]
    replacement = {"requested_title": "New title", "page_title": "New title", "page_id": 11, "revision_id": 12, "raw_html": "article"}

    corrected_questions, corrected_pages, corrections = apply_title_corrections(
        questions, pages, {"Old title": replacement}
    )

    assert corrected_questions == [{"question_id": 7, "wikipedia_title": "New title", "question": "Who directed Film?"}]
    assert corrected_pages == [replacement]
    assert corrections == [{"question_id": 7, "old_title": "Old title", "new_title": "New title", "old_page_id": 10, "new_page_id": 11}]
    assert questions[0]["wikipedia_title"] == "Old title"
