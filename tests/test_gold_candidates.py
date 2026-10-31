from scripts.gold_candidates import normalize_alias, rank_question_candidates


def test_alias_normalization_uses_casefolded_token_text():
    assert normalize_alias("  New\u00a0York-City! ") == "new york city"


def test_candidates_are_page_scoped_and_ranked_by_alias_then_relation_then_proximity():
    question = {
        "question_id": 7,
        "wikipedia_title": "Example Page",
        "subject": "Example Page",
        "relation": "director",
        "accepted_answers": ["Jane Doe", "J. Doe"],
    }
    chunks = [
        {
            "chunk_id": "wrong",
            "page_id": "other",
            "page_title": "Other Page",
            "chunk_index": 0,
            "text": "Jane Doe was the director.",
        },
        {
            "chunk_id": "one",
            "page_id": "example",
            "page_title": "Example Page",
            "chunk_index": 0,
            "text": "Jane Doe appears in this paragraph, far from Example Page.",
        },
        {
            "chunk_id": "two",
            "page_id": "example",
            "page_title": "Example Page",
            "chunk_index": 1,
            "text": "Example Page director Jane Doe directed Example Page.",
        },
        {
            "chunk_id": "three",
            "page_id": "example",
            "page_title": "Example Page",
            "chunk_index": 2,
            "text": "Jane Doe and Jane Doe are mentioned, but the relation is absent.",
        },
    ]

    candidates = rank_question_candidates(question, chunks)

    assert [candidate["chunk_id"] for candidate in candidates] == ["three", "two", "one"]
    assert all(candidate["page_id"] == "example" for candidate in candidates)
    assert candidates[0]["alias_occurrences"] == 2
    assert candidates[1]["relation_occurrences"] == 1
    assert candidates[1]["matched_aliases"] == ["Jane Doe"]


def test_candidates_mark_aliases_without_claiming_relation_support():
    question = {
        "question_id": 8,
        "wikipedia_title": "Example Page",
        "subject": "Example Page",
        "relation": "occupation",
        "accepted_answers": ["Artist"],
    }
    chunks = [{
        "chunk_id": "c1",
        "page_id": "p1",
        "page_title": "Example Page",
        "chunk_index": 0,
        "text": "Artist appeared in the cast list.",
    }]

    candidate = rank_question_candidates(question, chunks)[0]

    assert candidate["matched_aliases"] == ["Artist"]
    assert candidate["relation_occurrences"] == 0
    assert candidate["relational_support"] is False


def test_candidates_match_conservative_short_form_of_accepted_alias():
    question = {
        "question_id": 9,
        "wikipedia_title": "Example Page",
        "subject": "Example Page",
        "relation": "father",
        "accepted_answers": ["Peter III of Russia"],
    }
    chunks = [{
        "chunk_id": "c1",
        "page_id": "p1",
        "page_title": "Example Page",
        "chunk_index": 0,
        "text": "Example Page was the son of Peter III, emperor of Russia.",
    }]

    candidates = rank_question_candidates(question, chunks)

    assert len(candidates) == 1
    assert candidates[0]["matched_aliases"] == ["Peter III of Russia"]


def test_candidates_match_small_adjectival_gap_in_genre_phrase():
    question = {
        "question_id": 10,
        "wikipedia_title": "Example Film",
        "subject": "Example Film",
        "relation": "genre",
        "accepted_answers": ["comedy film"],
    }
    chunks = [{
        "chunk_id": "c1",
        "page_id": "p1",
        "page_title": "Example Film",
        "chunk_index": 0,
        "text": "Example Film is an American comedy drama film.",
    }]

    assert rank_question_candidates(question, chunks)[0]["matched_aliases"] == ["comedy film"]
