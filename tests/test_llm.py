import pytest

from LLM import DEFAULT_MODEL, get_llm


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_llm()


def test_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert get_llm().model_name == DEFAULT_MODEL


def test_model_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")

    assert get_llm().model_name == "some/other-model"


def test_model_argument_beats_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GROQ_MODEL", "from/env")

    assert get_llm(model="from/arg").model_name == "from/arg"

