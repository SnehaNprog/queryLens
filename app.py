import streamlit as st

from Loaders import pdfloader, webloader
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

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about your sources...")

if question:
    # Store and display the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Temporary assistant placeholder
    placeholder = "Answering comes in step 24."

    st.session_state.messages.append({
        "role": "assistant",
        "content": placeholder,
    })

    with st.chat_message("assistant"):
        st.markdown(placeholder)
