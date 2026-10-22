from pathlib import Path

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
