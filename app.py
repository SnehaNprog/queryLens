import streamlit as st

from Loaders import pdfloader, webloader
from QAChain import stream_answer
from Retriever import Retriever

st.set_page_config(page_title="QueryLens", page_icon="🧠", layout="wide")


@st.cache_resource
def get_retriever():
    """Create the Retriever once per server process and share it across re-runs.

    Without this cache, every click would re-run the script and rebuild the
    retriever (and reload the 2 GB models). cache_resource keeps ONE object alive.
    """
    return Retriever()


retriever = get_retriever()

st.title("QueryLens 🧠")
st.caption("Ask questions about webpages and PDFs in any language.")

# ---- Sidebar: add sources ----
with st.sidebar:
    st.header("Add sources")

    url_tab, pdf_tab = st.tabs(["🌐 Webpage", "📄 PDF"])

    with url_tab:
        url = st.text_input("Page URL", placeholder="https://en.wikipedia.org/wiki/...")
        add_url = st.button("Add webpage", use_container_width=True)

        if add_url and url.strip():
            with st.spinner("Indexing webpage..."):
                added = retriever.add(webloader(url.strip()))

            if added > 0:
                st.success(f"Added {added} chunks.")
            else:
                st.info("This page is already loaded.")

    with pdf_tab:
        pdf_file = st.file_uploader("Upload a PDF", type="pdf")
        add_pdf = st.button("Add PDF", use_container_width=True)

        if add_pdf and pdf_file is not None:
            with st.spinner("Indexing PDF..."):
                added = retriever.add(pdfloader(pdf_file, source_name=pdf_file.name))

            if added > 0:
                st.success(f"Added {added} chunks.")
            else:
                st.info("This PDF is already loaded.")

    st.divider()
    st.subheader("Loaded sources")
    # This sits AFTER the button handlers on purpose: Streamlit runs top to bottom,
    # so a source added in this run already shows up in the list.
    loaded_sources = retriever.sources()
    if loaded_sources:
        for source in loaded_sources:
            st.markdown(f"- {source}")
    else:
        st.caption("Nothing loaded yet.")

# ---- Main area: chat ----


def render_sources(sources, cited):
    """Show the passages behind an answer: the cited ones first, then the rest."""
    cited_sources = [s for s in sources if s["id"] in cited]
    other_sources = [s for s in sources if s["id"] not in cited]

    with st.expander(f"Sources ({len(cited_sources)} cited)"):
        for s in cited_sources:
            page = f", page {s['page']}" if s["page"] is not None else ""
            st.markdown(f"**[{s['id']}] {s['source']}{page}**")
            st.markdown(f"> {s['text']}")

        if other_sources:
            st.caption("Other retrieved passages")
            for s in other_sources:
                page = f", page {s['page']}" if s["page"] is not None else ""
                st.caption(f"[{s['id']}] {s['source']}{page}: {s['text'][:200]}")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit re-runs this script on every interaction, so redraw the saved chat.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"], message["cited"])

question = st.chat_input("Ask a question about your sources...")

if question:
    # History for follow-ups: everything said BEFORE this question.
    history = [(m["role"], m["content"]) for m in st.session_state.messages]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not retriever.sources():
        notice = "Add a webpage or PDF in the sidebar first, then ask your question."
        st.session_state.messages.append({"role": "assistant", "content": notice})
        with st.chat_message("assistant"):
            st.markdown(notice)
    else:
        captured = {"sources": [], "cited": []}

        def answer_tokens():
            """Yield answer text as it is generated; remember sources and citations."""
            for kind, payload in stream_answer(question, retriever, history=history):
                if kind == "sources":
                    captured["sources"] = payload
                elif kind == "token":
                    yield payload
                else:
                    captured["cited"] = payload["cited"]

        with st.chat_message("assistant"):
            answer = st.write_stream(answer_tokens())
            render_sources(captured["sources"], captured["cited"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": captured["sources"],
            "cited": captured["cited"],
        })
