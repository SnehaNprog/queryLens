import groq
import httpx
import pytest
import requests
from pypdf.errors import PdfReadError

from querylens.errors import NoTextError, llm_error_message, source_error_message

REQUEST = httpx.Request("POST", "https://api.groq.com/test")


def groq_status_error(error_class, status):
    return error_class("boom", response=httpx.Response(status, request=REQUEST), body=None)


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (requests.exceptions.MissingSchema("x"), "valid URL"),
        (requests.exceptions.Timeout(), "too long"),
        (requests.exceptions.ConnectionError(), "Couldn't reach"),
        (http_error(404), "404"),
        (http_error(403), "refused access"),
        (http_error(500), "500"),
        (PdfReadError("bad"), "Couldn't read that PDF"),
        (NoTextError("No text found."), "No text found."),
        (ValueError("weird"), "ValueError"),
    ],
)
def test_source_error_messages(exc, expected):
    assert expected in source_error_message(exc)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("GROQ_API_KEY is not set."), "GROQ_API_KEY is not set."),
        (groq_status_error(groq.AuthenticationError, 401), "rejected the API key"),
        (groq_status_error(groq.RateLimitError, 429), "rate limit"),
        (groq_status_error(groq.NotFoundError, 404), "GROQ_MODEL"),
        (groq.APIConnectionError(request=REQUEST), "Couldn't reach Groq"),
        (groq.APITimeoutError(request=REQUEST), "Couldn't reach Groq"),
        (KeyError("x"), "KeyError"),
    ],
)
def test_llm_error_messages(exc, expected):
    assert expected in llm_error_message(exc)
