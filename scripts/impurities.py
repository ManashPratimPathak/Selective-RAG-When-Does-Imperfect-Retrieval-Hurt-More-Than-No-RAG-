"""Adversarial distractor generators and helpers for Selective RAG stress-testing.

Provides builders for three structured retrieval impurity categories:
1. Entity-Overlap Distractors: Passages that share words/names with the question subject
   without answering it, testing lexical confusion and retriever false-confidence.
2. Relation-Matched Distractors: Passages describing other entities of the same relation
   with competing answers, testing semantic type-attraction.
3. Temporal / Conflicting Distractors: Passages presenting historical, former, or conflicting
   facts with explicit temporal markers, testing context-memory conflict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from .tokenization import tokenize
except ImportError:
    from tokenization import tokenize


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "who", "which", "where", "when",
}


def extract_entity_tokens(text: str) -> list[str]:
    """Extract informative alphanumeric entity tokens ignoring common stopwords."""
    raw_tokens = tokenize(text)
    return [t.lower() for t in raw_tokens if t.lower() not in STOPWORDS and len(t) > 1]


def contains_any_answer(text: str, accepted_answers: Sequence[str]) -> bool:
    """Check if any accepted answer appears as a substring in text."""
    lowered = text.lower()
    for ans in accepted_answers:
        ans_clean = str(ans).strip().lower()
        if len(ans_clean) >= 2 and ans_clean in lowered:
            return True
    return False


def build_temporal_conflict_passage(
    subject: str,
    relation: str,
    conflicting_answer: str,
    *,
    seed_offset: int = 0,
) -> str:
    """Construct an encyclopedic temporal distractor passage with historical/superseded markers."""
    templates = {
        "director": (
            f"During early production and pre-release development of {subject}, "
            f"{conflicting_answer} was originally attached as the principal director before "
            f"subsequent creative reorganization."
        ),
        "producer": (
            f"Initial production registries for {subject} credited {conflicting_answer} as the "
            f"primary producer during preliminary studio development."
        ),
        "screenwriter": (
            f"Earlier draft revisions and development treatments for {subject} were originally "
            f"scripted by screenwriter {conflicting_answer} prior to final rewrites."
        ),
        "author": (
            f"Early literary archives and syndicated serializations originally attributed the manuscript "
            f"of {subject} to {conflicting_answer} before canonical attribution."
        ),
        "composer": (
            f"During initial recording sessions and provisional arrangements of {subject}, "
            f"{conflicting_answer} was credited as the lead musical composer."
        ),
        "capital": (
            f"Historically, prior to the modern administrative capital reorganization, "
            f"{conflicting_answer} served as the former historical capital and seat of government of {subject}."
        ),
        "capital of": (
            f"In historical treaties prior to modern administrative consolidation, "
            f"{conflicting_answer} was documented as the historical territory or state encompassing {subject}."
        ),
        "country": (
            f"Under historical pre-modern territorial boundaries, the region of {subject} was "
            f"formerly administered as part of {conflicting_answer} prior to contemporary sovereignty."
        ),
        "genre": (
            f"During its initial debut and retrospective critique, {subject} was widely categorized "
            f"under the classification of {conflicting_answer} before later stylistic reassessment."
        ),
        "occupation": (
            f"Prior to achieving widespread prominence, {subject} was historically recorded "
            f"as practicing primarily as a professional {conflicting_answer}."
        ),
        "place of birth": (
            f"Earlier municipal records and historical census filings previously noted {conflicting_answer} "
            f"as the recorded place of residence and origin for {subject}."
        ),
        "sport": (
            f"In earlier amateur athletic circuits prior to professional specialization, "
            f"{subject} actively competed in {conflicting_answer}."
        ),
        "religion": (
            f"Historical biographical accounts note that during early upbringing, "
            f"{subject} was raised in the tradition of {conflicting_answer} before later affiliation."
        ),
        "father": (
            f"Genealogical archives and former historical registries previously recorded "
            f"{conflicting_answer} as the documented paternal guardian of {subject}."
        ),
        "mother": (
            f"Genealogical archives and former historical registries previously recorded "
            f"{conflicting_answer} as the documented maternal guardian of {subject}."
        ),
    }
    fallback = (
        f"Historical records and earlier documentation regarding {subject} previously "
        f"associated the entity with {conflicting_answer} as its documented {relation}."
    )
    return templates.get(relation.lower().strip(), fallback)


def build_relation_passage(subject: str, relation: str, answer: str) -> str:
    """Build a standard factual passage describing a competing entity for the same relation."""
    relation_phrasings = {
        "director": f"{subject} is a recognized film directed by {answer}.",
        "producer": f"{subject} is a notable production produced by {answer}.",
        "screenwriter": f"The screenplay and writing for {subject} was authored by {answer}.",
        "author": f"{subject} is a literary work authored by {answer}.",
        "composer": f"The musical composition and score for {subject} was created by {answer}.",
        "capital": f"The official designated capital city of {subject} is {answer}.",
        "capital of": f"{subject} is historically designated as the capital city of {answer}.",
        "country": f"{subject} is a recognized geographic territory located in {answer}.",
        "genre": f"{subject} is classified primarily within the musical or creative genre of {answer}.",
        "occupation": f"{subject} is a notable public figure known for working as a {answer}.",
        "place of birth": f"{subject} was born and spent formative years in {answer}.",
        "sport": f"{subject} is an athlete who professionally competes in {answer}.",
        "religion": f"{subject} is historically affiliated with the faith and practice of {answer}.",
        "father": f"Historical and biographical records identify {answer} as the father of {subject}.",
        "mother": f"Historical and biographical records identify {answer} as the mother of {subject}.",
    }
    fallback = f"According to factual documentation, the {relation} of {subject} is {answer}."
    return relation_phrasings.get(relation.lower().strip(), fallback)
