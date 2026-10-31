from scripts.verify_passages import build_verification_row, finalize_verification_row, relation_supported_candidate, supplemental_candidate, carry_forward_review, promote_candidate


def test_verification_row_selects_two_distinct_candidates_and_leaves_relation_for_review():
    question = {
        "question_id": 1,
        "question": "Who directed Example?",
        "accepted_answers": ["Jane Doe"],
    }
    candidates = [
        {"chunk_id": "c1", "candidate_rank": 1, "text": "Jane Doe appears here."},
        {"chunk_id": "c1", "candidate_rank": 2, "text": "Duplicate candidate."},
        {"chunk_id": "c2", "candidate_rank": 3, "text": "Example was directed by Jane Doe."},
    ]

    row = build_verification_row(question, candidates)

    assert row["clean_doc_1"] == "[c1] Jane Doe appears here."
    assert row["clean_doc_2"] == "[c2] Example was directed by Jane Doe."
    assert row["alias_present"] == {"clean_doc_1": True, "clean_doc_2": True}
    assert row["relation_supported"] is None
    assert row["status"] == "needs_manual_review"


def test_verification_row_rejects_when_two_candidate_passages_are_unavailable():
    question = {"question_id": 2, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"]}

    row = build_verification_row(question, [])

    assert row["clean_doc_1"] is None
    assert row["clean_doc_2"] is None
    assert row["status"] == "rejected_no_candidate_passages"
    assert row["reviewer_note"]


def test_automated_screen_does_not_reject_answer_only_candidates_without_review():
    question = {"question_id": 3, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"]}
    row = build_verification_row(
        question,
        [
            {"chunk_id": "c1", "candidate_rank": 1, "matched_aliases": ["Jane Doe"], "text": "Jane Doe appears here."},
            {"chunk_id": "c2", "candidate_rank": 2, "matched_aliases": ["Jane Doe"], "text": "Jane Doe appears again."},
        ],
    )

    final = finalize_verification_row(row, relation_supported=False)

    assert final["automated_relation_match"] is False
    assert final["relation_supported"] is None
    assert final["ambiguous"] is None
    assert final["status"] == "needs_manual_review"


def test_relation_support_accepts_directed_by_answer_wording():
    candidate = {
        "relation_keyword": "director",
        "matched_aliases": ["Jane Doe"],
        "text": "Example was directed by Jane Doe in 2010.",
    }

    assert relation_supported_candidate(candidate)


def test_relation_support_accepts_fuzzy_genre_alias_with_descriptor():
    candidate = {
        "relation_keyword": "genre",
        "matched_aliases": ["comedy film"],
        "text": "The film is an American comedy drama film directed by Jane Doe.",
    }

    assert relation_supported_candidate(candidate)


def test_verification_prefers_a_separate_chunk_over_an_overlapping_sentence():
    question = {"question_id": 4, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"]}
    candidates = [
        {"chunk_id": "c1", "candidate_rank": 1, "text": "Example was directed by Jane Doe. More context follows."},
        {"chunk_id": "c1:sentence_00", "candidate_rank": 2, "text": "Example was directed by Jane Doe."},
        {"chunk_id": "c2", "candidate_rank": 3, "text": "Example was released in 2010."},
    ]

    row = build_verification_row(question, candidates)

    assert row["clean_doc_1"].startswith("[c1]")
    assert row["clean_doc_2"].startswith("[c2]")


def test_verification_uses_two_sentences_when_only_one_chunk_exists():
    question = {"question_id": 5, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"]}
    candidates = [
        {"chunk_id": "c1", "candidate_rank": 1, "text": "Example was directed by Jane Doe. It was released in 2010."},
        {"chunk_id": "c1:sentence_00", "candidate_rank": 2, "text": "Example was directed by Jane Doe."},
        {"chunk_id": "c1:sentence_01", "candidate_rank": 3, "text": "It was released in 2010."},
    ]

    row = build_verification_row(question, candidates)

    assert row["clean_doc_1"].startswith("[c1:sentence_00]")
    assert row["clean_doc_2"].startswith("[c1:sentence_01]")


def test_supplemental_sentence_can_establish_the_relation():
    question = {"accepted_answers": ["Jane Doe"], "relation": "director"}
    chunk = {"chunk_id": "c1", "text": "Example was directed by Jane Doe. It was released in 2010."}

    candidate = supplemental_candidate(chunk, question, "c1:sentence_00", "Example was directed by Jane Doe.", 1)

    assert relation_supported_candidate(candidate)


def test_alias_presence_checks_selected_text_not_candidate_metadata():
    question = {"question_id": 6, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"]}
    candidates = [
        {"chunk_id": "c1", "candidate_rank": 1, "matched_aliases": ["Jane Doe"], "text": "Example was directed by Jane Doe."},
        {"chunk_id": "c2", "candidate_rank": 2, "matched_aliases": ["Jane Doe"], "text": "The film was released in 2010."},
    ]

    row = build_verification_row(question, candidates)

    assert row["alias_present"] == {"clean_doc_1": True, "clean_doc_2": False}


def test_completed_review_survives_regeneration_only_when_evidence_matches():
    generated = {"question_id": 1, "question": "Who directed Example?", "accepted_answers": ["Jane Doe"], "clean_doc_1": "[c1] Directed by Jane Doe.", "clean_doc_2": "[c2] Another passage.", "status": "needs_manual_review"}
    previous = {**generated, "status": "verified_manual", "reviewer_note": "Read both passages."}

    assert carry_forward_review(generated, previous) == previous
    assert carry_forward_review({**generated, "clean_doc_2": "[c3] Changed passage."}, previous)["status"] == "needs_manual_review"


def test_review_override_promotes_a_direct_evidence_chunk():
    candidates = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]

    assert promote_candidate(candidates, "c2")[0]["chunk_id"] == "c2"
