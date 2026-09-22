"""Fetch and freeze the Wikipedia pages referenced by the PopQA sample."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


API_ROOT = "https://en.wikipedia.org/w/rest.php/v1/page"
DEFAULT_USER_AGENT = "SelectiveRAG-PopQA/0.1 (research corpus; contact: project-maintainer)"


def build_page_url(title: str) -> str:
    return f"{API_ROOT}/{quote(str(title), safe='')}/with_html"


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
        if response.status_code != 200:
            outcome["error"] = response.text[:500]
            return None, outcome
        payload = response.json()
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
