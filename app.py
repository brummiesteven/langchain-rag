"""Streamlit UI for the RAG chat application.

Provides a chat interface for asking questions about uploaded documents.
Documents are ingested via the sidebar, and answers are generated using
the two-stage RAG chain (condense + retrieve + generate).
"""

import tempfile
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("RAG Chat")

# --- Check if backend is configured ---
# Attempt to load settings; if env vars are missing, degrade gracefully
# so the UI can still be previewed without Azure credentials.
try:
    from config import get_settings

    get_settings()
    _configured = True
except EnvironmentError:
    _configured = False


def _get_chain():
    """Lazy-import the RAG chain to avoid loading heavy modules until needed."""
    from rag.chain import get_rag_chain

    return get_rag_chain()


# --- Session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = _get_chain() if _configured else None

# --- Sidebar ---
with st.sidebar:
    if not _configured:
        st.warning(
            "Azure credentials not configured. "
            "Copy `.env.example` to `.env` and fill in your values, then restart."
        )
        st.divider()

    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents to chat with",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=not _configured,
    )
    if uploaded_files and st.button("Ingest Documents", disabled=not _configured):
        from ingestion.ingest import SPLITTER, SUPPORTED_EXTENSIONS, load_document
        from rag.vectorstore import get_vector_store

        with st.spinner("Ingesting documents..."):
            vector_store = get_vector_store()
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                if suffix.lower() not in SUPPORTED_EXTENSIONS:
                    st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                    continue

                # Write to a temp file because document loaders need a path on disk
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    docs = load_document(tmp_path)
                    chunks = SPLITTER.split_documents(docs)
                    vector_store.add_documents(chunks)
                    st.success(
                        f"Ingested {uploaded_file.name} ({len(chunks)} chunks)"
                    )
                finally:
                    # Clean up the temp file after ingestion
                    tmp_path.unlink(missing_ok=True)

    st.divider()
    if st.button("Clear Chat History"):
        from rag.chain import clear_session_history

        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['name']}**")
                    st.caption(src["snippet"])

# --- Chat input ---
if not _configured:
    st.chat_input("Ask a question about your documents", disabled=True)
elif prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        st.markdown(answer)

        # Show which chunks were retrieved so the user can verify the answer
        sources = []
        if source_documents:
            with st.expander(f"Sources ({len(source_documents)})"):
                for i, doc in enumerate(source_documents, 1):
                    # Try source path first, fall back to title, then generic label
                    name = doc.metadata.get(
                        "source", doc.metadata.get("title", f"Chunk {i}")
                    )
                    # Show first 200 chars as a preview snippet
                    snippet = doc.page_content[:200] + (
                        "..." if len(doc.page_content) > 200 else ""
                    )
                    st.markdown(f"**{i}. {name}**")
                    st.caption(snippet)
                    sources.append({"name": name, "snippet": snippet})

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
