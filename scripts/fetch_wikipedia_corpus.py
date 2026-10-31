"""Fetch and freeze the Wikipedia pages referenced by the PopQA sample."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


API_ROOT = "https://en.wikipedia.org/w/rest.php/v1/page"
WIKI_ROOT = "https://en.wikipedia.org/wiki"
DEFAULT_USER_AGENT = "SelectiveRAG-PopQA/0.1 (research corpus; contact: project-maintainer)"


def build_page_url(title: str) -> str:
    # Preserve the requested PopQA title. Following redirects can silently
    # freeze an unrelated article under the wrong question/page mapping.
    return f"{API_ROOT}/{quote(str(title), safe='')}/with_html?redirect=no"


def build_fallback_url(title: str) -> str:
    return f"{WIKI_ROOT}/{quote(str(title).replace(' ', '_'), safe='()/:')}?redirect=no"


def unresolved_current_failures(failures: list[dict[str, Any]], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unresolved failures whose requested title is absent now."""
    current_titles = {str(page.get("requested_title")) for page in pages}
    return [
        failure
        for failure in failures
        if failure.get("requested_title") not in current_titles
    ]


def page_has_substantive_content(page: dict[str, Any]) -> bool:
    soup = BeautifulSoup(page.get("raw_html", ""), "html.parser")
    main = soup.select_one(".mw-parser-output")
    text = (main or soup).get_text(" ", strip=True)
    folded = " ".join(text.casefold().split())
    if "this page is a redirect" in folded:
        return False
    return len(folded) >= 250


def cached_page_matches_request(page: dict[str, Any], title: str) -> bool:
    if page.get("requested_title") != title or not page_has_substantive_content(page):
        return False
    if page.get("page_title") == title:
        return True
    source = urlparse(str(page.get("source_url") or ""))
    if source.netloc != "en.wikipedia.org" or not source.path.startswith("/wiki/"):
        return False
    fetched_title = unquote(source.path.removeprefix("/wiki/")).replace("_", " ")
    return fetched_title == title


def choose_backup(
    backup_pool: pd.DataFrame, *, popularity_group: str, relation: str, used_ids: set[Any]
) -> dict[str, Any] | None:
    relation_column = "relation" if "relation" in backup_pool.columns else "prop"
    id_column = "question_id" if "question_id" in backup_pool.columns else "id"
    candidates = backup_pool[
        (backup_pool["popularity_group"] == popularity_group)
        & (backup_pool[relation_column] == relation)
        & (~backup_pool[id_column].isin(used_ids))
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0].to_dict()


def fetch_page(
    session: requests.Session, title: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 30
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    requested_url = build_page_url(title)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        response = session.get(
            requested_url,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            allow_redirects=True,
        )
        outcome = {
            "requested_title": title,
            "requested_url": requested_url,
            "final_url": response.url,
            "status_code": response.status_code,
            "retrieved_at": retrieved_at,
            "retry_after": response.headers.get("Retry-After"),
        }
        if response.status_code == 429:
            # The REST API may throttle research clients independently of the
            # regular article endpoint.  The article HTML contains stable
            # article/revision identifiers in Wikimedia's page config, so use
            # it as a rate-limit-safe fallback rather than waiting through
            # repeated REST retries.
            fallback_url = build_fallback_url(title)
            response = session.get(
                fallback_url,
                headers={"User-Agent": user_agent, "Accept": "text/html"},
                timeout=timeout,
                allow_redirects=False,
            )
            outcome.update(
                {
                    "fallback_url": fallback_url,
                    "fallback_status_code": response.status_code,
                    "final_url": response.url,
                }
            )
            if response.status_code == 200:
                html = response.text
                article_id = _metadata_value(html, "wgArticleId")
                revision_id = _metadata_value(html, "wgCurRevisionId") or _metadata_value(html, "wgRevisionId")
                soup = BeautifulSoup(html, "html.parser")
                heading = soup.find("h1", id="firstHeading")
                final_title = heading.get_text(" ", strip=True) if heading else title
                page = {
                    "page_title": final_title,
                    "page_id": article_id,
                    "revision_id": revision_id,
                    "revision_timestamp": None,
                    "retrieved_at": retrieved_at,
                    "source_url": response.url,
                    "requested_title": title,
                    "license": "CC BY-SA 4.0",
                    "raw_html": html,
                }
                if article_id and revision_id and html and page_has_substantive_content(page):
                    outcome.update({"status": "ok", "status_code": 200, "page_title": final_title, "revision_id": revision_id, "fallback": True})
                    return page, outcome
                direct_url = f"{WIKI_ROOT}/{quote(str(title).replace(' ', '_'), safe='()/:')}"
                direct_response = session.get(
                    direct_url,
                    headers={"User-Agent": user_agent, "Accept": "text/html"},
                    timeout=timeout,
                    allow_redirects=False,
                )
                if direct_response.status_code == 200:
                    html = direct_response.text
                    article_id = _metadata_value(html, "wgArticleId")
                    revision_id = _metadata_value(html, "wgCurRevisionId") or _metadata_value(html, "wgRevisionId")
                    soup = BeautifulSoup(html, "html.parser")
                    heading = soup.find("h1", id="firstHeading")
                    final_title = heading.get_text(" ", strip=True) if heading else title
                    page = {
                        "page_title": final_title,
                        "page_id": article_id,
                        "revision_id": revision_id,
                        "revision_timestamp": None,
                        "retrieved_at": retrieved_at,
                        "source_url": direct_response.url,
                        "requested_title": title,
                        "license": "CC BY-SA 4.0",
                        "raw_html": html,
                    }
                    if article_id and revision_id and page_has_substantive_content(page):
                        outcome.update({"status": "ok", "status_code": 200, "page_title": final_title, "revision_id": revision_id, "fallback": True, "direct_fallback": True})
                        return page, outcome
            response = response
        if response.status_code != 200:
            direct_url = f"{WIKI_ROOT}/{quote(str(title).replace(' ', '_'), safe='()/:')}"
            direct_response = session.get(
                direct_url,
                headers={"User-Agent": user_agent, "Accept": "text/html"},
                timeout=timeout,
                allow_redirects=False,
            )
            if direct_response.status_code == 200:
                html = direct_response.text
                article_id = _metadata_value(html, "wgArticleId")
                revision_id = _metadata_value(html, "wgCurRevisionId") or _metadata_value(html, "wgRevisionId")
                soup = BeautifulSoup(html, "html.parser")
                heading = soup.find("h1", id="firstHeading")
                final_title = heading.get_text(" ", strip=True) if heading else title
                page = {
                    "page_title": final_title,
                    "page_id": article_id,
                    "revision_id": revision_id,
                    "revision_timestamp": None,
                    "retrieved_at": retrieved_at,
                    "source_url": direct_response.url,
                    "requested_title": title,
                    "license": "CC BY-SA 4.0",
                    "raw_html": html,
                }
                if article_id and revision_id and page_has_substantive_content(page):
                    outcome.update({"status": "ok", "status_code": 200, "page_title": final_title, "revision_id": revision_id, "direct_fallback": True})
                    return page, outcome
            outcome["error"] = response.text[:500]
            return None, outcome
        try:
            payload = response.json()
        except ValueError:
            # Some upstream 200 responses are HTML error shells; the
            # substantive-content branch below will try the article URL.
            payload = {}
        latest = payload.get("latest") or {}
        page = {
            "page_title": payload.get("title") or payload.get("key") or title,
            "page_id": payload.get("id"),
            "revision_id": latest.get("id"),
            "revision_timestamp": latest.get("timestamp"),
            "retrieved_at": retrieved_at,
            "source_url": response.url,
            "requested_title": title,
            "license": payload.get("license"),
            "raw_html": payload.get("html", ""),
        }
        if not page_has_substantive_content(page):
            direct_url = f"{WIKI_ROOT}/{quote(str(title).replace(' ', '_'), safe='()/:')}"
            direct_response = session.get(
                direct_url,
                headers={"User-Agent": user_agent, "Accept": "text/html"},
                timeout=timeout,
                allow_redirects=False,
            )
            if direct_response.status_code == 200:
                html = direct_response.text
                article_id = _metadata_value(html, "wgArticleId")
                revision_id = _metadata_value(html, "wgCurRevisionId") or _metadata_value(html, "wgRevisionId")
                soup = BeautifulSoup(html, "html.parser")
                heading = soup.find("h1", id="firstHeading")
                final_title = heading.get_text(" ", strip=True) if heading else title
                direct_page = {
                    "page_title": final_title,
                    "page_id": article_id,
                    "revision_id": revision_id,
                    "revision_timestamp": None,
                    "retrieved_at": retrieved_at,
                    "source_url": direct_response.url,
                    "requested_title": title,
                    "license": "CC BY-SA 4.0",
                    "raw_html": html,
                }
                if article_id and revision_id and page_has_substantive_content(direct_page):
                    outcome.update({
                        "status": "ok", "status_code": 200, "page_title": final_title,
                        "revision_id": revision_id, "direct_fallback": True,
                        "final_url": direct_response.url,
                    })
                    return direct_page, outcome
            outcome.update({"status": "unusable_page", "error": "redirect_or_insufficient_content"})
            return None, outcome
        if not page["page_id"] or not page["revision_id"] or not page["raw_html"]:
            outcome["error"] = "response missing page id, revision id, or html"
            return None, outcome
        outcome.update({"status": "ok", "page_title": page["page_title"], "revision_id": page["revision_id"]})
        return page, outcome
    except (requests.RequestException, ValueError) as error:
        return None, {
            "requested_title": title,
            "requested_url": requested_url,
            "retrieved_at": retrieved_at,
            "status": "failed",
            "error": str(error),
        }


def _metadata_value(html: str, key: str) -> str | None:
    """Read a numeric value from Wikimedia's embedded page configuration."""
    marker = f'"{key}":'
    start = html.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = start
    while end < len(html) and html[end].isdigit():
        end += 1
    return html[start:end] or None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
