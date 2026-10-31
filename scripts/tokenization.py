"""Shared lexical tokenization for retrieval documents and questions."""

from __future__ import annotations

import unicodedata


def tokenize(text: str) -> list[str]:
    """Lowercase text, separate non-letters/numbers, and return its tokens."""
    normalized = "".join(
        char if unicodedata.category(char)[0] in {"L", "N"} else " "
        for char in text.lower()
    )
    return normalized.split()
