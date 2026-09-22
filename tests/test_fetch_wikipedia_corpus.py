import pandas as pd

from scripts.fetch_wikipedia_corpus import (
    build_page_url,
    choose_backup,
)


def test_build_page_url_quotes_title_and_uses_rest_endpoint():
    assert build_page_url("The Express: The Ernie Davis Story") == (
        "https://en.wikipedia.org/w/rest.php/v1/page/The%20Express%3A%20The%20Ernie%20Davis%20Story/with_html"
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
