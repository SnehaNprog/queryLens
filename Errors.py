import groq
import requests
from pypdf.errors import PyPdfError


class NoTextError(Exception):
    """A source loaded fine but contained no readable text."""


def source_error_message(exc):
    """A friendly message for anything that can go wrong while loading a source."""
    if isinstance(exc, NoTextError):
        return str(exc)

    if isinstance(exc, (requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema,
                        requests.exceptions.InvalidURL)):
        return "That doesn't look like a valid URL. Enter a full address starting with http:// or https://."
    if isinstance(exc, requests.exceptions.Timeout):
        return "The page took too long to respond. Try again, or try a different page."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Couldn't reach that address. Check the URL and your internet connection."
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            return "That page wasn't found (error 404). Check the URL."
        if status in (401, 403):
            return f"The site refused access (error {status}). It may block automated readers or need a login."
        return f"The site returned an error ({status}). Try again later."

    if isinstance(exc, PyPdfError):
        return "Couldn't read that PDF. It may be corrupted or password-protected."

    return f"Something went wrong while loading the source: {type(exc).__name__}."


def llm_error_message(exc):
    """A friendly message for anything that can go wrong while asking the model."""
    if isinstance(exc, RuntimeError):
        return str(exc)  # LLM.get_llm already explains a missing GROQ_API_KEY

    if isinstance(exc, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, groq.RateLimitError):
        return "Groq's rate limit was reached. Wait a moment and ask again."
    if isinstance(exc, groq.NotFoundError):
        return "Groq doesn't have that model. Set GROQ_MODEL in .env to a model your key can use."
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return "Couldn't reach Groq. Check your internet connection and try again."

    return f"The model request failed: {type(exc).__name__}."
