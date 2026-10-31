"""Canonical answer normalization used by scoring."""

from __future__ import annotations

import unicodedata


def normalize_answer(value: object) -> str:
    """Lowercase, remove punctuation/articles, and collapse whitespace."""
    text = str(value or "").lower()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


normalise_answer = normalize_answer


def normalize_answers(values: list[object]) -> list[str]:
    return [normalize_answer(value) for value in values]


normalise_answers = normalize_answers
