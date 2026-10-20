import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from LLM import get_llm
import re


SYSTEM_PROMPT = """You are a multilingual question-answering assistant.

Follow these rules strictly:

1. Answer ONLY using the numbered context passages provided.
   Never use your own knowledge, training data, or outside information.
2. If the context does not contain enough information to answer the
   question, clearly say that you do not know based on the provided context.
   Do not guess or make up information.
3. Always reply in the SAME LANGUAGE as the user's question,
   even when the context passages are written in another language.
4. When using information from a passage, cite its number using the
   format [1], [2], etc. Place citations near the claims they support.
5. Do not cite a passage unless your answer actually uses information
   from it.
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(results):
    """Turn [(chunk, score), ...] into one numbered string for the prompt."""
    lines = []

    for i, (chunk, score) in enumerate(results, start=1):
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page")

        location = f"({source}"
        if page is not None:
            location += f", page {page}"
        location += ")"

        lines.append(f"[{i}] {location} {chunk.page_content}")

    return "\n\n".join(lines)


def build_chain(llm=None):
    """prompt -> LLM -> plain string."""
    return PROMPT | (llm or get_llm()) | StrOutputParser()


def answer(question, retriever, llm=None, k=5):
    """Retrieve passages and generate an answer grounded in their content."""
    results = retriever.search(question, k=k)

    context = (
        format_context(results)
        if results
        else "(no passages found)"
    )

    chain = build_chain(llm)

    return chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )



def build_sources(results):
    """Turn retrieval results into numbered source dictionaries."""
    sources = []

    for i, (chunk, score) in enumerate(results, start=1):
        sources.append({
            "id": i,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "text": chunk.page_content,
        })

    return sources


def extract_citations(text):
    """Return sorted unique passage numbers cited in the answer."""
    citations = re.findall(r"\[(\d+)\]", text)
    return sorted(set(int(citation) for citation in citations))


def answer_with_sources(question, retriever, llm=None, k=5, history=None):
    """Return the answer, retrieved sources, and cited passage numbers."""
    # Search with the rewritten question, but answer the original one.
    search_query = rewrite_question(question, history, llm)
    results = retriever.search(search_query, k=k)

    context = (
        format_context(results)
        if results
        else "(no passages found)"
    )

    chain = build_chain(llm)

    answer_text = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer_text,
        "sources": build_sources(results),
        "cited": extract_citations(answer_text),
    }


def answer(question, retriever, llm=None, k=5, history=None):
    """Return only the generated answer."""
    return answer_with_sources(
        question,
        retriever,
        llm=llm,
        k=k,
        history=history,
    )["answer"]


def stream_answer(question, retriever, llm=None, k=5, history=None):
    """Stream answer events: sources, tokens, then done."""
    # Search with the rewritten question, but answer the original one.
    search_query = rewrite_question(question, history, llm)
    results = retriever.search(search_query, k=k)

    context = (
        format_context(results)
        if results
        else "(no passages found)"
    )

    # Send sources before generating the answer.
    yield ("sources", build_sources(results))

    pieces = []

    chain = build_chain(llm)

    for piece in chain.stream({
        "context": context,
        "question": question,
    }):
        pieces.append(piece)
        yield ("token", piece)

    full_text = "".join(pieces)

    yield (
        "done",
        {
            "answer": full_text,
            "cited": extract_citations(full_text),
        },
    )



# ---- Step 21b: conversation memory via query rewriting ----

REWRITE_SYSTEM_PROMPT = """You are a question-rewriting assistant.

Follow these rules strictly:

1. Rewrite the user's follow-up question into a STANDALONE question.
   Use the chat history to resolve references, pronouns, and missing context.
2. Always preserve the SAME LANGUAGE as the follow-up question.
   Do not translate it, even if the chat history uses another language.
3. Output ONLY the rewritten question.
   Do not include explanations, introductions, quotation marks, or labels.
4. If the question is already standalone and does not need chat history,
   return it unchanged.
"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "Chat history:\n{history}\n\nFollow-up question: {question}"),
    ]
)


def format_history(history, max_turns=6):
    """Turn [("user", text), ("assistant", text), ...] into a readable string.

    Example output:
        User: How do plants make energy?
        Assistant: They use photosynthesis [1].
    """
    recent = history[-max_turns:]

    lines = []

    for role, text in recent:
        lines.append(f"{role.capitalize()}: {text}")

    return "\n".join(lines)


def rewrite_question(question, history, llm=None):
    """Make a follow-up question standalone so retrieval can find the right passages."""
    # No history means nothing to resolve, so skip the LLM call.
    if not history:
        return question

    chain = REWRITE_PROMPT | (llm or get_llm()) | StrOutputParser()

    rewritten = chain.invoke({
        "history": format_history(history),
        "question": question,
    })

    rewritten = rewritten.strip()

    return rewritten if rewritten else question
