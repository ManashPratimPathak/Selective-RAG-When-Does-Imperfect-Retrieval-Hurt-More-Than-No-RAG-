from scripts.chunk_corpus import chunk_page


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count)) + "."


def test_chunk_page_tracks_sections_overlap_and_indexed_representation():
    record = {
        "page_id": "p1",
        "revision_id": "r1",
        "page_title": "Example Page",
        "provenance": {"source_url": "https://example.test/page"},
        "text": f"Example Page\n\n{words('intro', 80)}\n\nHistory\n\n{words('history', 80)}",
    }

    chunks = chunk_page(record, chunk_words=100, overlap_words=20, min_words=25)

    assert len(chunks) == 2
    assert chunks[0]["section_title"] == "Example Page"
    assert chunks[1]["section_title"] == "History"
    assert chunks[0]["word_count"] == len(chunks[0]["text"].split())
    assert chunks[1]["text"].split()[:20] == chunks[0]["text"].split()[-20:]
    assert chunks[0]["indexed_text"].startswith("Example Page\n\n")
    assert chunks[0]["text"] != chunks[0]["indexed_text"]
    assert chunks[0]["chunk_id"] == "p1_r1_0000"
    assert chunks[1]["source_url"] == "https://example.test/page"


def test_chunk_page_merges_a_short_final_tail():
    record = {
        "page_id": "p2",
        "revision_id": "r2",
        "page_title": "Short Tail",
        "provenance": {"source_url": "https://example.test/short"},
        "text": f"Short Tail\n\n{words('body', 100)}\n\n{words('tail', 10)}",
    }

    chunks = chunk_page(record, chunk_words=100, overlap_words=20, min_words=25)

    assert len(chunks) == 1
    assert chunks[0]["word_count"] >= 100


def test_chunk_page_keeps_short_factual_blocks():
    record = {
        "page_id": "p3",
        "revision_id": "r3",
        "page_title": "When in Rome",
        "text": "When in Rome\n\nThe film was released in 1952.\n\nProduced by Clarence Brown",
    }

    chunks = chunk_page(record)

    assert "Produced by Clarence Brown" in " ".join(chunk["text"] for chunk in chunks)
