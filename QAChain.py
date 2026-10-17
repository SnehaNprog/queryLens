from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from LLM import get_llm


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