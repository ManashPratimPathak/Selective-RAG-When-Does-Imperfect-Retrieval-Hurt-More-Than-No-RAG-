import pytest

from scripts.replacement_pool import candidate_from_popqa


def test_candidate_preserves_popqa_identity_and_stratum():
    source = {
        "id": 12, "subj": "Film A", "prop": "director", "obj": "Jane Doe",
        "question": "Who directed Film A?", "s_wiki_title": "Film A (film)",
        "s_pop": 90, "possible_answers": '["Jane Doe", "Wrong Person"]',
        "o_aliases": '["J. Doe"]',
    }

    candidate = candidate_from_popqa(source, q25=227, q75=5395.5)

    assert candidate["question_id"] == 12
    assert candidate["popularity_group"] == "tail"
    assert candidate["accepted_answers"] == ["Jane Doe", "J. Doe"]
    assert candidate["wikipedia_title"] == "Film A (film)"


def test_candidate_rejects_middle_popularity_and_time_dependent_questions():
    source = {
        "id": 12, "subj": "Film A", "prop": "director", "obj": "Jane Doe",
        "question": "Who directs Film A today?", "s_wiki_title": "Film A (film)",
        "s_pop": 90, "possible_answers": '["Jane Doe"]',
    }

    assert candidate_from_popqa(source, q25=227, q75=5395.5) is None
    assert candidate_from_popqa({**source, "question": "Who directed Film A?", "s_pop": 1000}, q25=227, q75=5395.5) is None


def test_candidate_excludes_external_identifiers_from_answer_aliases():
    source = {
        "id": 13, "subj": "Album A", "prop": "producer", "obj": "John Cale",
        "question": "Who produced Album A?", "s_wiki_title": "Album A",
        "s_pop": 90, "o_aliases": '["John Davies Cale", "00033982178 IPI"]',
    }

    candidate = candidate_from_popqa(source, q25=227, q75=5395.5)

    assert candidate["accepted_answers"] == ["John Cale", "John Davies Cale"]
