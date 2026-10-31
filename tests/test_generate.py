from scripts.generate import (
    USER_PROMPT_TEMPLATE,
    build_messages,
    build_prompt,
    generate_answer,
    looks_explanatory,
    parse_answer,
    prompt_contract_hash,
)


def test_build_prompt_uses_same_structure_for_all_conditions():
    question = "Who directed Example?"
    prompts = {
        condition: build_prompt(question, condition, documents)
        for condition, documents in (
            ("no_rag", []),
            ("clean_rag", ["verified passage 1", "verified passage 2"]),
            ("imperfect_rag", ["BM25 rank 1", "BM25 rank 2", "BM25 rank 3"]),
        )
    }

    assert "Documents:\nNo documents provided." in prompts["no_rag"]
    assert "Documents:\n[D1] verified passage 1\n[D2] verified passage 2" in prompts["clean_rag"]
    assert "[D1] BM25 rank 1\n[D2] BM25 rank 2\n[D3] BM25 rank 3" in prompts["imperfect_rag"]
    assert all("Question:\nWho directed Example?" in prompt for prompt in prompts.values())
    assert all(prompt.endswith("Shortest answer:") for prompt in prompts.values())


def test_generate_answer_uses_apply_chat_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return {"input_ids": [[10, 11]]}

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [12]
            assert skip_special_tokens is True
            return "Paris\nextra"

    class Model:
        device = "cpu"

        def generate(self, **inputs):
            assert inputs["input_ids"] == [[10, 11]]
            assert inputs["num_beams"] == 1
            assert inputs["use_cache"] is True
            assert inputs["max_new_tokens"] == 16
            assert inputs["do_sample"] is False
            return [[10, 11, 12]]

    tokenizer = Tokenizer()
    answer = generate_answer(tokenizer, Model(), "Question text")

    assert answer == "Paris"
    assert tokenizer.kwargs["add_generation_prompt"] is True


def test_generate_answer_refuses_to_truncate_prompt_or_instruction():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [[10, 11]]}

    class Model:
        device = "cpu"

    try:
        generate_answer(Tokenizer(), Model(), "Question text", max_new_tokens=1, max_input_tokens=2)
    except ValueError as error:
        assert "refuse to truncate" in str(error)
    else:
        raise AssertionError("expected context overflow to be rejected")


def test_parse_answer_extracts_only_the_first_generated_line():
    assert parse_answer("Gary Fleder\nHe directed the film.") == "Gary Fleder"
    assert parse_answer("  Gary Fleder  ") == "Gary Fleder"


def test_explanation_check_does_not_flag_long_name_lists():
    assert not looks_explanatory("Walter F. Parkes Laurie MacDonald Steve Starkey Robert Zemeckis")
    assert looks_explanatory("The answer is Gary Fleder because he directed the film.")


def test_prompt_contract_is_exact_and_hashable():
    assert USER_PROMPT_TEMPLATE == "Documents:\n{documents}\n\nQuestion:\n{question}\n\nShortest answer:"
    assert len(prompt_contract_hash()) == 64
