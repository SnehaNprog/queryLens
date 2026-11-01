from langchain_core.runnables import RunnableLambda

from querylens.ingestion.loaders import Document
from querylens.generation.qa_chain import (
    SYSTEM_PROMPT,
    answer,
    answer_with_sources,
    build_sources,
    extract_citations,
    format_context,
    format_history,
    rewrite_question,
    stream_answer,
)


class FakeRetriever:
    """Stands in for Retriever so tests need no models."""

    def __init__(self, results):
        self.results = results

    def search(self, query, k=5):
        return self.results


def make_fake_llm(reply="fake answer"):
    """Returns (llm, seen). seen collects every prompt sent to the LLM."""
    seen = []

    def fake(prompt_value):
        seen.append(prompt_value.to_string())
        return reply

    return RunnableLambda(fake), seen


def test_system_prompt_has_the_key_rules():
    prompt = SYSTEM_PROMPT.lower()

    assert "context" in prompt
    assert "language" in prompt
    assert "do not know" in prompt or "don't know" in prompt


def test_format_context_numbers_passages_and_shows_source():
    results = [
        (
            Document(
                "Plants make energy.",
                {"source": "bio.pdf", "page": 1}
            ),
            0.9,
        ),
        (
            Document(
                "Markets fell.",
                {"source": "https://x.test"}
            ),
            0.5,
        ),
    ]

    formatted = format_context(results)

    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "bio.pdf" in formatted
    assert "page 1" in formatted
    assert "https://x.test" in formatted

    second_passage = formatted.split("[2]", 1)[1]
    assert "page" not in second_passage


def test_answer_sends_context_and_question_to_llm():
    llm, seen = make_fake_llm("Paris")

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 2}
            ),
            0.9,
        )
    ])

    result = answer("What is the capital?", retriever, llm=llm)

    assert result == "Paris"
    assert len(seen) == 1
    assert "Paris is the capital of France." in seen[0]
    assert "What is the capital?" in seen[0]


def test_answer_without_results_still_calls_llm():
    llm, seen = make_fake_llm("I don't know.")

    result = answer("anything", FakeRetriever([]), llm=llm)

    assert result == "I don't know."
    assert len(seen) == 1
    assert "(no passages found)" in seen[0]
    assert "anything" in seen[0]

def test_build_sources_numbers_and_metadata():
    results = [
        (
            Document(
                "Plants use sunlight.",
                {"source": "bio.pdf", "page": 1}
            ),
            0.9,
        ),
        (
            Document(
                "Markets are changing.",
                {"source": "news.pdf"}
            ),
            0.5,
        ),
    ]

    sources = build_sources(results)

    assert len(sources) == 2

    assert sources[0]["id"] == 1
    assert sources[0]["text"] == "Plants use sunlight."
    assert sources[0]["source"] == "bio.pdf"
    assert sources[0]["page"] == 1

    assert sources[1]["id"] == 2
    assert sources[1]["text"] == "Markets are changing."
    assert sources[1]["source"] == "news.pdf"
    assert sources[1]["page"] is None


def test_extract_citations():
    text = "Plants use light [1]. Also see [3] and again [1]."

    assert extract_citations(text) == [1, 3]

    assert extract_citations("There are no citations here.") == []


def test_answer_with_sources_returns_dict():
    reply = "Paris is the capital [1]."
    llm, seen = make_fake_llm(reply)

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 2}
            ),
            0.9,
        )
    ])

    result = answer_with_sources(
        "What is the capital?",
        retriever,
        llm=llm,
    )

    assert result["answer"] == reply
    assert result["cited"] == [1]
    assert result["sources"][0]["source"] == "geo.pdf"


def test_answer_with_sources_when_nothing_found():
    llm, seen = make_fake_llm("I don't know.")

    result = answer_with_sources(
        "anything",
        FakeRetriever([]),
        llm=llm,
    )

    assert result["sources"] == []
    assert result["cited"] == []
    assert result["answer"] == "I don't know."
    assert len(seen) == 1


def make_streaming_llm(pieces):
    def fake(prompt_value):
        for piece in pieces:
            yield piece

    return RunnableLambda(fake)



def test_stream_answer_event_order():
    llm = make_streaming_llm([
        "Paris ",
        "is the ",
        "capital [1].",
    ])

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 1}
            ),
            0.9,
        )
    ])

    events = list(stream_answer("q?", retriever, llm=llm))

    kinds = [kind for kind, _ in events]

    assert kinds == [
        "sources",
        "token",
        "token",
        "token",
        "done",
    ]


def test_stream_answer_tokens_join_to_full_answer():
    llm = make_streaming_llm([
        "Paris ",
        "is the ",
        "capital [1].",
    ])

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 1}
            ),
            0.9,
        )
    ])

    events = list(stream_answer("q?", retriever, llm=llm))

    tokens = [
        payload
        for kind, payload in events
        if kind == "token"
    ]

    done = events[-1][1]

    assert "".join(tokens) == done["answer"]
    assert done["cited"] == [1]


def test_stream_answer_sources_come_first():
    llm = make_streaming_llm(["Paris [1]."])

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 1}
            ),
            0.9,
        )
    ])

    events = list(stream_answer("q?", retriever, llm=llm))

    assert events[0][0] == "sources"
    assert events[0][1][0]["source"] == "geo.pdf"


def test_stream_answer_is_lazy():
    calls = []

    def fake(prompt_value):
        calls.append(prompt_value.to_string())
        yield "Paris [1]."

    llm = RunnableLambda(fake)

    retriever = FakeRetriever([
        (
            Document(
                "Paris is the capital of France.",
                {"source": "geo.pdf", "page": 1}
            ),
            0.9,
        )
    ])

    stream = stream_answer("q?", retriever, llm=llm)

    # Nothing executes when the generator is created.
    assert calls == []

    # First iteration retrieves and yields sources.
    first_event = next(stream)

    assert first_event[0] == "sources"
    assert calls == []

    # The LLM has not run until iteration continues.
    second_event = next(stream)

    assert second_event == ("token", "Paris [1].")
    assert len(calls) == 1


# TODO M: test_format_history_labels_roles
def test_format_history_labels_roles():
    history = [("user", "Hi"), ("assistant", "Hello")]

    assert format_history(history) == "User: Hi\nAssistant: Hello"


# TODO N: test_format_history_keeps_only_recent_turns
def test_format_history_keeps_only_recent_turns():
    history = [
        ("user", "Message 1"),
        ("assistant", "Message 2"),
        ("user", "Message 3"),
        ("assistant", "Message 4"),
        ("user", "Message 5"),
        ("assistant", "Message 6"),
        ("user", "Message 7"),
        ("assistant", "Message 8"),
        ("user", "Message 9"),
        ("assistant", "Message 10"),
    ]

    result = format_history(history, max_turns=4)

    assert "Message 3" not in result
    assert "Message 10" in result
    assert result.splitlines() == [
        "User: Message 7",
        "Assistant: Message 8",
        "User: Message 9",
        "Assistant: Message 10",
    ]


# TODO O: test_rewrite_without_history_skips_the_llm
def test_rewrite_without_history_skips_the_llm():
    llm, seen = make_fake_llm()

    assert rewrite_question("q?", [], llm=llm) == "q?"
    assert rewrite_question("q?", None, llm=llm) == "q?"
    assert seen == []


# TODO P: test_rewrite_calls_llm_and_strips_result
def test_rewrite_calls_llm_and_strips_result():
    llm, seen = make_fake_llm("  What do plants release?  ")

    result = rewrite_question(
        "and what do they release?",
        [("user", "How do plants make energy?")],
        llm=llm,
    )

    assert result == "What do plants release?"
    assert "How do plants make energy?" in seen[0]


# TODO Q: test_rewrite_falls_back_when_model_returns_nothing
def test_rewrite_falls_back_when_model_returns_nothing():
    llm, seen = make_fake_llm("   ")

    question = "and what do they release?"

    assert rewrite_question(
        question,
        [("user", "How do plants make energy?")],
        llm=llm,
    ) == question


# TODO R: test_answer_searches_with_rewritten_question_but_answers_original
def test_answer_searches_with_rewritten_question_but_answers_original():
    class RecordingRetriever(FakeRetriever):
        def __init__(self, results):
            super().__init__(results)
            self.queries = []

        def search(self, query, k=5):
            self.queries.append(query)
            return super().search(query, k)

    retriever = RecordingRetriever([])
    llm, seen = make_fake_llm("What do plants release?")

    answer_with_sources(
        "and what do they release?",
        retriever,
        llm=llm,
        history=[("user", "How do plants make energy?")],
    )

    assert retriever.queries == ["What do plants release?"]
    assert "and what do they release?" in seen[-1]

    