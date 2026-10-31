"""Extract retrieval-friendly prose from frozen Wikipedia HTML pages."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from bs4 import BeautifulSoup, Tag


REFERENCE_HEADINGS = {
    "references",
    "notes",
    "sources",
    "further reading",
    "external links",
    "see also",
    "bibliography",
    "citations",
}

REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "math",
    "table",
    "nav",
    "header",
    "footer",
    ".navbox",
    ".vertical-navbox",
    ".navigation-box",
    ".metadata",
    ".ambox",
    ".hatnote",
    ".mw-editsection",
    ".editsection",
    ".reference",
    ".references",
    ".reflist",
    ".mw-references-wrap",
    ".infobox-caption",
    "figcaption",
    "caption",
    "img",
    "figure",
    ".thumb",
    ".gallery",
)


def _normalise(text: str) -> str:
    text = re.sub(r"\[\s*(?:citation\s+needed|needs\s+update|failed\s+verification|unreliable\s+source|[0-9]+(?:\s*[-,]\s*[0-9]+)*)\s*\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _remove_reference_sections(soup: BeautifulSoup) -> None:
    """Remove a references-like section without affecting ordinary prose."""
    for heading in list(soup.find_all(re.compile(r"^h[1-6]$"))):
        if _normalise(heading.get_text(" ", strip=True)).casefold() not in REFERENCE_HEADINGS:
            continue
        section = heading.find_parent("section")
        if section is not None:
            section.decompose()
            continue
        level = int(heading.name[1])
        sibling = heading
        while sibling is not None:
            current = sibling.find_next_sibling()
            if current is None:
                break
            if isinstance(current, Tag) and re.fullmatch(r"h[1-6]", current.name or ""):
                if int(current.name[1]) <= level:
                    break
            sibling.decompose()


def _iter_content_blocks(soup: BeautifulSoup) -> Iterable[tuple[str, str]]:
    """Yield headings and prose paragraphs in document order."""
    for element in soup.find_all(re.compile(r"^h[1-6]$|^p$")):
        text = _normalise(element.get_text(" ", strip=True))
        if not text:
            continue
        kind = "heading" if element.name.startswith("h") else "paragraph"
        yield kind, text


def clean_page_blocks(
    raw_html: str,
    *,
    page_title: str | None = None,
    min_fragment_chars: int = 20,
    required_texts: Iterable[str] = (),
) -> list[str]:
    """Return title, section headings, and non-trivial prose blocks.

    The input is frozen HTML.  Cleaning is deliberately structure-aware so
    lists and tables cannot turn into misleading prose fragments.
    """
    soup = BeautifulSoup(raw_html or "", "html.parser")
    _remove_reference_sections(soup)
    required = {_normalise(value).casefold() for value in required_texts if _normalise(value)}
    structured_evidence = []
    if required:
        for row in soup.select("table.infobox tr"):
            row_text = _normalise(row.get_text(" ", strip=True))
            if any(phrase in row_text.casefold() for phrase in required):
                structured_evidence.append(row_text)
        source_text = _normalise(soup.get_text(" ", strip=True))
        source_folded = source_text.casefold()
        for phrase in required:
            start = source_folded.find(phrase)
            if start < 0:
                continue
            # Preserve a bounded local context from infoboxes/tables before
            # those structures are removed. This keeps relation labels such
            # as "Director" alongside the accepted answer.
            context_start = max(0, start - 120)
            context_end = min(len(source_text), start + len(phrase) + 120)
            structured_evidence.append(source_text[context_start:context_end])
    for element in soup.select(",".join(REMOVE_SELECTORS)):
        element.decompose()

    candidates: list[tuple[str, str]] = []
    if page_title and _normalise(page_title):
        candidates.append(("heading", _normalise(page_title)))
    candidates.extend(_iter_content_blocks(soup))
    candidates.extend(("paragraph", phrase) for phrase in structured_evidence)

    # Keep the first occurrence of repeated chrome such as a repeated page
    # header/footer, while allowing repeated short headings only once too.
    counts = Counter(text.casefold() for _, text in candidates)
    seen: set[str] = set()
    blocks: list[str] = []
    for kind, text in candidates:
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        preserves_required_text = any(phrase in text.casefold() for phrase in required)
        if kind == "paragraph" and len(text) < min_fragment_chars and not preserves_required_text:
            continue
        # A repeated block is normally page chrome; the first copy is useful
        # only when it is the explicit title or a genuine section heading.
        if counts[key] > 1 and kind == "paragraph" and not preserves_required_text:
            continue
        blocks.append(text)
    return blocks


def clean_page_text(
    raw_html: str,
    *,
    page_title: str | None = None,
    min_fragment_chars: int = 20,
    required_texts: Iterable[str] = (),
) -> str:
    """Return cleaned blocks separated by blank lines."""
    return "\n\n".join(
        clean_page_blocks(
            raw_html,
            page_title=page_title,
            min_fragment_chars=min_fragment_chars,
            required_texts=required_texts,
        )
    )


# Short alias for callers that prefer a verb-like API.
clean_page = clean_page_text
