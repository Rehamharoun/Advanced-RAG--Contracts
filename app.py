"""Chat demo UI for the contract Q&A system.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from advanced_rag import config  # noqa: E402
from advanced_rag.pipeline import build_retriever  # noqa: E402

st.set_page_config(page_title="Contract Q&A", page_icon="📄", layout="centered")
st.title("📄 Contract Q&A")
st.caption("Advanced RAG: hybrid retrieval + knowledge graph. Answers are grounded in the contracts and cite the source file.")


@st.cache_resource(show_spinner="Building index from PDFs (first run only)...")
def load_retriever():
    retriever, docs, _ = build_retriever(config.PDF_DIR)
    return retriever, sorted(docs)


try:
    retriever, doc_names = load_retriever()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Loaded documents")
    for name in doc_names:
        st.write("•", name)
    st.divider()
    st.header("Try asking")
    st.write("• What is the liability cap in the social media integration agreement?")
    st.write("• How long does Meta keep data after an account deletion request?")
    st.write("• How much notice is needed to terminate the Juhayna PepsiCo distribution agreement?")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources, contexts):
    with st.expander("Sources and retrieved context"):
        st.write("**Top documents:**", ", ".join(dict.fromkeys(sources)))
        for c in contexts:
            st.text(c)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"], m["contexts"])

if question := st.chat_input("Ask a question about the contracts..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching contracts..."):
            res = retriever.ask(question)
        st.markdown(res["answer"])
        render_sources(res["sources"], res["contexts"])

    st.session_state.messages.append({
        "role": "assistant", "content": res["answer"],
        "sources": res["sources"], "contexts": res["contexts"],
    })
