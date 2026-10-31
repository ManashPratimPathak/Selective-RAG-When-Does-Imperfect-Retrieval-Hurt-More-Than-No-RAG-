"""Unit tests for adversarial distractor generators and impurity stress testing."""

from __future__ import annotations

import pytest

from scripts.impurities import (
    build_relation_passage,
    build_temporal_conflict_passage,
    contains_any_answer,
    extract_entity_tokens,
)


def test_extract_entity_tokens_filters_stopwords_and_short_words():
    text = "The quick brown fox of the river"
    tokens = extract_entity_tokens(text)
    assert "the" not in tokens
    assert "of" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "river" in tokens


def test_contains_any_answer_case_insensitive():
    text = "The movie was directed by Christopher Nolan in 2010."
    assert contains_any_answer(text, ["christopher nolan"])
    assert contains_any_answer(text, ["Nolan"])
    assert not contains_any_answer(text, ["Steven Spielberg", "James Cameron"])


def test_build_temporal_conflict_passage_supported_relations():
    director_doc = build_temporal_conflict_passage("Inception", "director", "Quentin Tarantino")
    assert "Inception" in director_doc
    assert "Quentin Tarantino" in director_doc
    assert "originally attached as the principal director" in director_doc

    capital_doc = build_temporal_conflict_passage("Serbia", "capital", "Kragujevac")
    assert "Serbia" in capital_doc
    assert "Kragujevac" in capital_doc
    assert "former historical capital" in capital_doc


def test_build_temporal_conflict_passage_fallback():
    custom_doc = build_temporal_conflict_passage("Custom Entity", "unknown_rel", "Alt Value")
    assert "Custom Entity" in custom_doc
    assert "Alt Value" in custom_doc
    assert "unknown_rel" in custom_doc


def test_build_relation_passage():
    film_doc = build_relation_passage("Interstellar", "director", "Christopher Nolan")
    assert "Interstellar" in film_doc
    assert "Christopher Nolan" in film_doc
    assert "directed by" in film_doc

    geo_doc = build_relation_passage("France", "capital", "Paris")
    assert "France" in geo_doc
    assert "Paris" in geo_doc
    assert "capital city" in geo_doc
