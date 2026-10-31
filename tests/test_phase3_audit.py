from scripts.phase3_audit import audit_phase3


def test_phase3_audit_requires_manual_review_and_two_distinct_passages():
    questions = [{"question_id": i, "question": f"Who wrote Article {i}?", "wikipedia_title": f"Article {i}"} for i in range(240)]
    gold = [{"requested_title": f"Article {i}", "page_id": i + 1000, "revision_id": 8, "source_url": "https://example.test"} for i in range(240)]
    distractors = [{"page_id": 999, "revision_id": 10, "source_url": "https://example.test"}]
    article_text = "Substantive article body with enough context to review the fact and find a second passage. Additional background follows."
    cleaned = [{"page_id": str(i + 1000), "role": "gold", "text": article_text} for i in range(240)] + [{"page_id": "999", "role": "distractor", "text": "A distractor page."}]
    chunks = [{"page_id": str(i + 1000), "chunk_id": f"{i}_8_{j:04d}", "text": "Article was written by Jane Doe." if j == 0 else "The article was published in 2000."} for i in range(240) for j in range(2)] + [{"page_id": "999", "chunk_id": "999_10_0000", "text": "A distractor page."}]
    review = [{
        "question_id": i,
        "question": f"Who wrote Article {i}?",
        "clean_doc_1": f"[{i}_8_0000] Article was written by Jane Doe.",
        "clean_doc_2": f"[{i}_8_0001] The article was published in 2000.",
        "relation_supported": True,
        "ambiguous": False,
        "reviewer_note": "Checked both passages.",
        "status": "verified_manual" if i else "verified_conservative_protocol",
    } for i in range(240)]
    question_review = [{"question_id": i, "question": f"Who wrote Article {i}?", "reviewer_status": "accepted"} for i in range(240)]

    audit = audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)

    assert audit["status"] == "incomplete"
    assert audit["manual_verified_questions"] == 239
    review[0]["status"] = "verified_manual"
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["status"] == "complete"
    review[0]["question"] = "Outdated question"
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["checks"]["passage_question_alignment"] is False
    review[0]["question"] = questions[0]["question"]
    question_review[0]["question"] = "Outdated question"
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["checks"]["accepted_question_review_coverage"] is False
    question_review[0]["question"] = questions[0]["question"]
    review[0]["clean_doc_2"] = "[0_8_0000:sentence_01] Overlapping sentence from the first chunk."
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["manual_verified_questions"] == 239
    review[0]["clean_doc_2"] = "[0_8_0001] A separate passage."
    review[0]["clean_doc_1"] = "[0_8_0000] Fabricated evidence."
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["manual_verified_questions"] == 239
    review[0]["clean_doc_1"] = "[0_8_0000] Article was written by Jane Doe."
    review[0]["clean_doc_2"] = "[0_8_0001] The article was published in 2000."
    cleaned[0]["text"] = "Article 0 This page is a redirect."
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["checks"]["gold_articles_substantive"] is False
    cleaned[0]["text"] = article_text
    question_review[0]["question_id"] = -1
    assert audit_phase3(questions, gold, distractors, cleaned, chunks, review, question_review, expected_distractors=1)["checks"]["accepted_question_review_coverage"] is False
