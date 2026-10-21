from pathlib import Path

from streamlit.testing.v1 import AppTest

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