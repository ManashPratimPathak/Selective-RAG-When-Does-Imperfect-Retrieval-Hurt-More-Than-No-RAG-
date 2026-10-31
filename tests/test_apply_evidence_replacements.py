from scripts.apply_evidence_replacements import apply_replacements, rebalance_distractors


def test_replacement_keeps_slot_stratum_and_swaps_only_unused_old_page():
    questions = [{"question_id": 1, "split": "test", "popularity_group": "tail", "relation": "producer", "wikipedia_title": "Old"}]
    gold = [{"requested_title": "Old", "page_id": 10}]
    candidate = {"question_id": 2, "split": "backup", "popularity_group": "tail", "relation": "producer", "wikipedia_title": "New"}
    page = {"requested_title": "New", "page_id": 11, "revision_id": 20}

    updated_questions, updated_pages = apply_replacements(questions, gold, {2: candidate}, {"New": page}, {1: 2})

    assert updated_questions == [{**candidate, "split": "test"}]
    assert updated_pages == [page]
    assert questions[0]["question_id"] == 1


def test_replacement_reclarifies_new_question_and_drops_old_source_question():
    questions = [{
        "question_id": 1, "split": "test", "popularity_group": "tail", "relation": "genre",
        "wikipedia_title": "Old (album)", "question": "What genre is Old (album)?",
        "source_question": "What genre is Old?", "subject": "Old", "sampling_seed": 42,
    }]
    candidate = {
        "question_id": 2, "split": "backup", "popularity_group": "tail", "relation": "genre",
        "wikipedia_title": "New (band)", "question": "What genre is New?",
        "subject": "New", "sampling_seed": 42,
    }
    page = {"requested_title": "New (band)", "page_id": 11, "revision_id": 20}

    updated, _ = apply_replacements(questions, [], {2: candidate}, {"New (band)": page}, {1: 2})

    assert updated[0]["question"] == "What genre is New (band)?"
    assert updated[0]["source_question"] == "What genre is New?"


def test_promoted_distractor_is_replaced_with_unused_revisioned_tail_page():
    promoted = {"requested_title": "Gold", "page_id": 11, "revision_id": 1}
    ordinary = {"requested_title": "Ordinary", "page_id": 12, "revision_id": 2}
    reserve = {"requested_title": "Reserve", "page_id": 13, "revision_id": 3}
    candidates = {7: {"question_id": 7, "popularity_group": "tail", "wikipedia_title": "Reserve"}}

    updated, substitutions = rebalance_distractors(
        [promoted, ordinary], [promoted], candidates, {"Reserve": reserve},
        {"11": "tail", "12": "head"},
    )

    assert updated == [ordinary, reserve]
    assert substitutions == [{"promoted_page_id": "11", "replacement_page_id": "13", "replacement_title": "Reserve", "entity_group": "tail", "source_question_id": 7}]
