import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.freeze_cleaned_corpus import clean_page_pair


def test_distractor_does_not_require_second_baseline_parse():
    calls = []

    def cleaner(html, *, page_title, required_texts=()):
        calls.append((page_title, tuple(required_texts)))
        return "Clean text"

    text, baseline = clean_page_pair("<article/>", "Example", "distractor", [], cleaner=cleaner)

    assert (text, baseline) == ("Clean text", None)
    assert calls == [("Example", ())]


def test_gold_keeps_baseline_for_answer_preservation_check():
    calls = []

    def cleaner(html, *, page_title, required_texts=()):
        calls.append(tuple(required_texts))
        return "Clean text"

    clean_page_pair("<article/>", "Example", "gold", ["answer"], cleaner=cleaner)

    assert calls == [("answer",), ()]
