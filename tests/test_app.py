from pathlib import Path

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from Loaders import Document

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def run_app():
    """Run app.py headlessly (no browser). AppTest gives us its widgets to inspect."""
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()



def test_app_starts_without_errors():
    at = run_app()

    assert not at.exception


def test_title_is_shown():
    at = run_app()

    assert at.title[0].value == "QueryLens 🧠"


def test_sidebar_has_add_buttons():
    at = run_app()

    labels = [button.label for button in at.sidebar.button]

    assert labels == ["Add webpage", "Add PDF"]



def test_question_is_added_to_chat_history():
    at = run_app()

    at.chat_input[0].set_value("What is AI?").run()

    messages = at.session_state.messages

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is AI?"
    assert len(messages) == 2


def test_chat_history_survives_a_second_question():
    at = run_app()

    at.chat_input[0].set_value("What is AI?").run()
    at.chat_input[0].set_value("What is ML?").run()

    messages = at.session_state.messages

    assert len(messages) == 4
    assert messages[0]["content"] == "What is AI?"
    assert messages[2]["content"] == "What is ML?"


# ---- Step 23: wiring sources to the retriever ----
# The real Retriever loads 2 GB models, so these tests use a fake one.

class FakeRetriever:
    def __init__(self):
        self.added = []

    def add(self, docs):
        new = [d for d in docs if d.metadata["source"] not in self.sources()]
        self.added.extend(new)
        return len(new)

    def sources(self):
        return sorted({d.metadata["source"] for d in self.added})


def run_app_with(monkeypatch, fake):
    st.cache_resource.clear()  # forget any retriever cached by an earlier test
    monkeypatch.setattr("Retriever.Retriever", lambda: fake)
    monkeypatch.setattr(
        "Loaders.webloader",
        lambda url: [Document("Hello", {"source": url, "type": "web"})],
    )
    return run_app()


def test_adding_a_webpage_indexes_it_and_lists_the_source(monkeypatch):
    fake = FakeRetriever()
    at = run_app_with(monkeypatch, fake)

    at.text_input[0].set_value("http://x.test")
    at.sidebar.button[0].click().run()

    assert len(fake.added) == 1
    assert "- http://x.test" in [m.value for m in at.sidebar.markdown]
    assert len(at.sidebar.success) == 1


def test_adding_same_page_twice_shows_info_not_success(monkeypatch):
    fake = FakeRetriever()
    at = run_app_with(monkeypatch, fake)

    at.text_input[0].set_value("http://x.test")
    at.sidebar.button[0].click().run()
    at.sidebar.button[0].click().run()

    assert len(fake.added) == 1
    assert len(at.sidebar.success) == 0
    assert "already loaded" in at.sidebar.info[0].value


def test_empty_url_does_nothing(monkeypatch):
    fake = FakeRetriever()
    at = run_app_with(monkeypatch, fake)

    at.sidebar.button[0].click().run()

    assert fake.added == []
    assert len(at.sidebar.success) == 0
    assert len(at.sidebar.info) == 0


# ---- Step 24: streaming answers and the sources panel ----

SOURCES = [
    {"id": 1, "source": "a.pdf", "page": 2, "text": "Paris is the capital."},
    {"id": 2, "source": "b.pdf", "page": None, "text": "Unrelated passage."},
]


class ChatRetriever:
    """Just enough Retriever for the chat: knows which sources are loaded."""

    def __init__(self, sources):
        self._sources = sources

    def sources(self):
        return self._sources


def run_chat_app(monkeypatch, loaded_sources=("a.pdf",)):
    """Run the app with a fake retriever and a fake stream_answer.

    Returns (app_test, calls); `calls` records every stream_answer invocation.
    """
    calls = []

    def fake_stream_answer(question, retriever, llm=None, k=5, history=None):
        calls.append({"question": question, "history": history})
        yield ("sources", SOURCES)
        for piece in ["Paris ", "is the capital ", "[1]."]:
            yield ("token", piece)
        yield ("done", {"answer": "Paris is the capital [1].", "cited": [1]})

    st.cache_resource.clear()
    monkeypatch.setattr("Retriever.Retriever", lambda: ChatRetriever(list(loaded_sources)))
    monkeypatch.setattr("QAChain.stream_answer", fake_stream_answer)
    return run_app(), calls


def test_asking_without_sources_explains_what_to_do(monkeypatch):
    at, calls = run_chat_app(monkeypatch, loaded_sources=())

    at.chat_input[0].set_value("What is AI?").run()

    assert "Add a webpage or PDF" in at.session_state.messages[1]["content"]
    assert calls == []


def test_answer_is_streamed_and_stored_with_its_sources(monkeypatch):
    at, calls = run_chat_app(monkeypatch)

    at.chat_input[0].set_value("Capital of France?").run()

    reply = at.session_state.messages[1]
    assert reply["role"] == "assistant"
    assert reply["content"] == "Paris is the capital [1]."
    assert reply["sources"] == SOURCES
    assert reply["cited"] == [1]
    assert len(calls) == 1


def test_sources_panel_counts_cited_passages(monkeypatch):
    at, _ = run_chat_app(monkeypatch)

    at.chat_input[0].set_value("Capital of France?").run()

    assert len(at.expander) == 1
    assert at.expander[0].label == "Sources (1 cited)"
    assert "**[1] a.pdf, page 2**" in [m.value for m in at.expander[0].markdown]


def test_history_is_passed_for_follow_up_questions(monkeypatch):
    at, calls = run_chat_app(monkeypatch)

    at.chat_input[0].set_value("Capital of France?").run()
    at.chat_input[0].set_value("And its population?").run()

    assert calls[0]["history"] == []
    assert calls[1]["history"] == [
        ("user", "Capital of France?"),
        ("assistant", "Paris is the capital [1]."),
    ]


def test_old_answers_keep_their_sources_after_rerun(monkeypatch):
    at, _ = run_chat_app(monkeypatch)

    at.chat_input[0].set_value("Capital of France?").run()
    at.chat_input[0].set_value("And its population?").run()

    assert len(at.expander) == 2


# ---- Step 25: error handling ----

def test_unreachable_page_shows_friendly_error(monkeypatch):
    fake = FakeRetriever()
    at = run_app_with(monkeypatch, fake)

    def unreachable(url):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr("Loaders.webloader", unreachable)
    at.text_input[0].set_value("http://nowhere.test")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "Couldn't reach" in at.sidebar.error[0].value
    assert fake.added == []
    assert len(at.sidebar.success) == 0


def test_page_without_text_shows_friendly_error(monkeypatch):
    fake = FakeRetriever()
    at = run_app_with(monkeypatch, fake)

    monkeypatch.setattr(
        "Loaders.webloader", lambda url: [Document("   ", {"source": url, "type": "web"})]
    )
    at.text_input[0].set_value("http://empty.test")
    at.sidebar.button[0].click().run()

    assert "no readable text" in at.sidebar.error[0].value
    assert fake.added == []


def test_model_failure_shows_error_and_keeps_the_conversation_clean(monkeypatch):
    at, _ = run_chat_app(monkeypatch)
    calls = []

    def flaky_stream_answer(question, retriever, llm=None, k=5, history=None):
        calls.append(history)
        if len(calls) == 1:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file.")
        yield ("sources", SOURCES)
        yield ("token", "Paris.")
        yield ("done", {"answer": "Paris.", "cited": [1]})

    monkeypatch.setattr("QAChain.stream_answer", flaky_stream_answer)

    at.chat_input[0].set_value("First question").run()

    assert not at.exception
    assert "GROQ_API_KEY is not set" in at.error[0].value
    assert at.session_state.messages[-1]["error"] is True

    at.chat_input[0].set_value("Second question").run()

    # The failed attempt must not leak into the history sent to the model.
    assert calls[1] == [("user", "First question")]
    assert at.session_state.messages[-1]["content"] == "Paris."
