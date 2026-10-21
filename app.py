import streamlit as st

st.set_page_config(page_title="QueryLens", page_icon="🧠", layout="wide")

st.title("QueryLens 🧠")
st.caption("Ask questions about webpages and PDFs in any language.")

# ---- Sidebar: add sources ----
with st.sidebar:
    st.header("Add sources")

    url_tab, pdf_tab = st.tabs(["🌐 Webpage", "📄 PDF"])

    with url_tab:
        url = st.text_input("Page URL", placeholder="https://en.wikipedia.org/wiki/...")
        add_url = st.button("Add webpage", use_container_width=True)

    with pdf_tab:
        pdf_file = st.file_uploader("Upload a PDF", type="pdf")
        add_pdf = st.button("Add PDF", use_container_width=True)

    st.divider()
    st.subheader("Loaded sources")
    st.caption("Nothing loaded yet.")  # replaced by the real list in step 23

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
