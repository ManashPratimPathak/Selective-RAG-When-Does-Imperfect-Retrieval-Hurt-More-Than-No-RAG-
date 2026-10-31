from scripts.clean_wikipedia import clean_page_blocks, clean_page_text


def test_cleaner_keeps_title_headings_and_prose_but_removes_page_chrome():
    html = """
    <html><body>
      <nav>Article Tools Home Login</nav>
      <table class="infobox"><tr><td>Director: Someone</td></tr></table>
      <h1>Example Page</h1>
      <h2>History</h2>
      <p>This is an ordinary paragraph with enough context to retain.</p>
      <p>tiny</p>
      <div class="navbox"><p>Related pages that are navigation.</p></div>
      <h2>References</h2><ol><li>Source one</li></ol>
      <p>Repeated footer text</p><p>Repeated footer text</p>
    </body></html>
    """

    blocks = clean_page_blocks(html, page_title="Example Page")

    assert blocks == [
        "Example Page",
        "History",
        "This is an ordinary paragraph with enough context to retain.",
    ]
    assert "Director" not in clean_page_text(html)


def test_cleaner_removes_reference_section_until_next_heading():
    html = "<h2>References</h2><p>citation list</p><h2>After</h2><p>Useful prose follows here.</p>"

    assert clean_page_blocks(html, page_title="Page") == ["Page", "After", "Useful prose follows here."]


def test_cleaner_can_preserve_a_brief_gold_answer_fragment():
    html = "<p>John Smith</p><p>This is a longer ordinary paragraph with useful context.</p>"

    assert "John Smith" in clean_page_text(html, required_texts=["John Smith"])


def test_cleaner_preserves_context_around_gold_answer_in_removed_structure():
    html = "<table><tr><th>Director</th><td>Jane Doe</td></tr></table><p>Useful prose follows here.</p>"

    cleaned = clean_page_text(html, required_texts=["Jane Doe"])

    assert "Director" in cleaned
    assert "Jane Doe" in cleaned


def test_cleaner_keeps_complete_matching_infobox_row():
    html = """<table class="infobox">
    <tr><th>Directed by</th><td>King Vidor</td></tr>
    <tr><th>Written by</th><td>June Mathis (scenario) John McDermott (adaptation) James O'Hanlon (adaptation) Winchell Smith (play) King Vidor (scenario)</td></tr>
    </table><p>This film was released in 1923 to favorable reviews.</p>"""

    cleaned = clean_page_text(html, page_title="Three Wise Fools", required_texts=["King Vidor"])

    assert "Written by June Mathis (scenario) John McDermott (adaptation) James O'Hanlon (adaptation) Winchell Smith (play) King Vidor (scenario)" in cleaned
