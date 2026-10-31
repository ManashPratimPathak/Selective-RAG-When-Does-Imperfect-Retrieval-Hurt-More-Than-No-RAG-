import pandas as pd

from scripts.fetch_wikipedia_corpus import (
    build_page_url,
    build_fallback_url,
    choose_backup,
    unresolved_current_failures,
    page_has_substantive_content,
    fetch_page,
    cached_page_matches_request,
)


def test_build_page_url_quotes_title_and_uses_rest_endpoint():
    assert build_page_url("The Express: The Ernie Davis Story") == (
        "https://en.wikipedia.org/w/rest.php/v1/page/The%20Express%3A%20The%20Ernie%20Davis%20Story/with_html?redirect=no"
    )


def test_choose_backup_matches_group_and_relation_and_avoids_used_ids():
    pool = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prop": ["director", "director", "occupation", "director"],
            "popularity_group": ["head", "tail", "head", "head"],
        }
    )

    backup = choose_backup(pool, popularity_group="head", relation="director", used_ids={1})

    assert backup["id"] == 4


def test_build_fallback_url_disables_wikipedia_redirects():
    assert build_fallback_url("Laika (company)") == (
        "https://en.wikipedia.org/wiki/Laika_(company)?redirect=no"
    )


def test_historical_failure_is_resolved_by_a_current_frozen_page():
    failures = [{"status": "unresolved_corpus_failure", "question_id": 7, "requested_title": "Fixed Page"}]
    pages = [{"requested_title": "Fixed Page", "page_title": "Fixed Page", "page_id": "p7"}]

    assert unresolved_current_failures(failures, pages) == []


def test_page_content_gate_rejects_redirect_shells_but_keeps_real_pages():
    assert not page_has_substantive_content({"raw_html": "<p>This page is a redirect.</p>"})
    assert page_has_substantive_content({"raw_html": "<p>" + ("A substantive article sentence. " * 20) + "</p>"})


def test_fetch_page_recovers_rest_200_redirect_shell_from_canonical_article():
    class Response:
        def __init__(self, url, *, payload=None, html=""):
            self.url = url
            self.status_code = 200
            self.headers = {}
            self._payload = payload
            self.text = html

        def json(self):
            return self._payload

    class Session:
        def get(self, url, **kwargs):
            if "/w/rest.php/" in url:
                return Response(url, payload={
                    "id": 1, "title": "Requested Title", "latest": {"id": 2},
                    "html": "<p>This page is a redirect.</p>",
                })
            return Response(url, html=(
                '<script>"wgArticleId":123,"wgCurRevisionId":456</script>'
                '<h1 id="firstHeading">Canonical Title</h1>'
                '<div class="mw-parser-output"><p>'
                + "A substantial article sentence. " * 20
                + "</p></div>"
            ))

    page, outcome = fetch_page(Session(), "Requested Title")

    assert outcome["status"] == "ok"
    assert outcome["direct_fallback"] is True
    assert page["requested_title"] == "Requested Title"
    assert page["page_title"] == "Canonical Title"
    assert page["page_id"] == "123"


def test_fetch_page_recovers_rest_200_non_json_response_from_direct_article():
    class Response:
        status_code = 200
        headers = {}

        def __init__(self, url, *, html=""):
            self.url = url
            self.text = html

        def json(self):
            raise ValueError("not JSON")

    class Session:
        def get(self, url, **kwargs):
            if "/w/rest.php/" in url:
                return Response(url, html="temporary upstream page")
            return Response(url, html=(
                '<script>"wgArticleId":123,"wgCurRevisionId":456</script>'
                '<h1 id="firstHeading">Direct Article</h1>'
                '<div class="mw-parser-output"><p>'
                + "A substantial article sentence. " * 20
                + "</p></div>"
            ))

    page, outcome = fetch_page(Session(), "Direct Article")

    assert page["page_id"] == "123"
    assert outcome["direct_fallback"] is True


def test_cache_accepts_a_revisioned_canonical_article_fetched_from_requested_url():
    page = {
        "requested_title": "Frederick William IV of Prussia",
        "page_title": "Frederick William IV",
        "source_url": "https://en.wikipedia.org/wiki/Frederick_William_IV_of_Prussia",
        "raw_html": "<p>" + ("A substantial article sentence. " * 20) + "</p>",
    }

    assert cached_page_matches_request(page, "Frederick William IV of Prussia")
    assert not cached_page_matches_request({**page, "source_url": "https://en.wikipedia.org/wiki/HubSpot"}, "Frederick William IV of Prussia")
