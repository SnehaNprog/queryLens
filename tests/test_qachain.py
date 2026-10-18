from langchain_core.runnables import RunnableLambda

from Loaders import Document
from QAChain import (
    SYSTEM_PROMPT,
    answer,
    format_context,
    build_sources,
    extract_citations,
    answer_with_sources,
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


