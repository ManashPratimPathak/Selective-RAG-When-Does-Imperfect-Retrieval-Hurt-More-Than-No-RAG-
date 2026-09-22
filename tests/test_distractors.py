import pandas as pd

from scripts.distractors import build_distractor_candidates, choose_distractor_target, sample_distractor_pages


def test_distractor_expansion_uses_only_dev_recall_gate():
    assert choose_distractor_target(None) == 500
    assert choose_distractor_target(0.949) == 500
    assert choose_distractor_target(0.95) == 1000


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "subj": ["Answer", "Movie A", "Person B", "Movie C", "Person D"],
            "prop": ["director", "director", "occupation", "genre", "occupation"],
            "obj": ["Answer Director", "Director A", "Writer B", "Drama", "Writer D"],
            "s_wiki_title": ["Answer Page", "Movie A", "Person B", "Movie C", "Person D"],
            "s_pop": [10000, 7000, 100, 200, 150],
        }
    )


def test_candidates_exclude_selected_test_entities_and_filter_relations():
    selected = pd.DataFrame(
        {
            "question_id": [1],
            "wikipedia_title": ["Answer Page"],
            "relation": ["director"],
        }
    )
    candidates = build_distractor_candidates(_source(), selected, relations={"director", "occupation"})

    assert "Answer Page" not in set(candidates["page_title"])
    assert set(candidates["entity_group"]) == {"head", "tail"}


def test_sample_is_deterministic_unique_and_contains_both_roles():
    selected = pd.DataFrame({"question_id": [1], "wikipedia_title": ["Answer Page"], "relation": ["director"]})
    candidates = build_distractor_candidates(_source(), selected)

    first = sample_distractor_pages(candidates, count=4, seed=42)
    second = sample_distractor_pages(candidates, count=4, seed=42)

    assert first.equals(second)
    assert first["page_title"].is_unique
    assert set(first["entity_group"]) == {"head", "tail"}
