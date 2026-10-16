import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_llm(model=None, temperature=0.0):
    """Build a Groq chat model. The key comes from GROQ_API_KEY in .env.

    The model can be changed with the GROQ_MODEL env var or the `model` argument,
    because Groq retires and adds models over time.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file: GROQ_API_KEY=your_key"
        )
    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        api_key=api_key,
    )
