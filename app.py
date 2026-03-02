"""Streamlit UI for the RAG chat application.

This is the main entry point of the application. It provides:
  - A chat interface where users type questions and get AI-generated answers
  - A sidebar for uploading documents (PDF, TXT, MD) to chat with
  - Source citations showing which document chunks informed each answer

HOW STREAMLIT WORKS (for those unfamiliar):
  Streamlit is a Python framework for building web UIs. The key thing to know
  is that Streamlit RE-RUNS THIS ENTIRE FILE from top to bottom on every user
  interaction (button click, text input, etc.). This means:
  - You can't store state in regular Python variables (they'd reset each run)
  - Instead, you use st.session_state, which persists across reruns
  - Expensive operations (like building the RAG chain) should be done once
    and cached in st.session_state

PAGE FLOW:
  1. On first load: check if Azure credentials are configured
  2. If not configured: show a warning, disable inputs (degraded mode)
  3. If configured: build the RAG chain (once), show chat interface
  4. User uploads docs via sidebar → ingested into Azure AI Search
  5. User types a question → sent through the RAG chain → answer displayed
  6. Chat history maintained per session (unique UUID per browser tab)
"""

import tempfile
import uuid
from pathlib import Path

import streamlit as st

# Configure the browser tab title and icon (must be the first Streamlit call)
st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("RAG Chat")

# ─── Check if backend is configured ──────────────────────────────────────────
# Try to load settings from env vars. If any required var is missing,
# get_settings() raises EnvironmentError and we enter "degraded mode"
# where the UI loads but all inputs are disabled.
# This lets someone preview the UI without needing Azure credentials.
try:
    from config import get_settings

    get_settings()
    _configured = True
except EnvironmentError:
    _configured = False


def _get_chain():
    """Lazy-import the RAG chain to avoid loading heavy modules until needed.

    LangChain and its dependencies are large. By importing inside this function
    (instead of at the top of the file), we avoid the import cost on every
    Streamlit rerun. The chain is only built once and cached in session_state.
    """
    from rag.chain import get_rag_chain

    return get_rag_chain()


# ─── Session State Initialisation ─────────────────────────────────────────────
# st.session_state is a dict-like object that persists across Streamlit reruns.
# We use it to store:
#   - session_id: unique ID for this browser tab's chat session (for history)
#   - messages: list of message dicts for rendering the chat display
#   - chain: the built RAG chain (expensive to create, so cached here)

if "session_id" not in st.session_state:
    # Generate a unique ID for this session. Used to key chat history so
    # different browser tabs have independent conversations.
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message is a dict: {"role": "user"|"assistant", "content": "...", "sources": [...]}
    # This list is used to render the chat display on each rerun.
    st.session_state.messages = []

if "chain" not in st.session_state:
    # Build the chain once and cache it. If credentials aren't configured,
    # store None so the UI can check and disable inputs accordingly.
    st.session_state.chain = _get_chain() if _configured else None

# ─── Sidebar: Document Upload + Controls ──────────────────────────────────────
with st.sidebar:
    # Show a warning if credentials aren't set up
    if not _configured:
        st.warning(
            "Azure credentials not configured. "
            "Copy `.env.example` to `.env` and fill in your values, then restart."
        )
        st.divider()

    st.header("Document Upload")

    # File uploader widget — accepts PDF, TXT, and MD files.
    # accept_multiple_files=True allows batch upload.
    # disabled=True when not configured (greyed out, can't interact).
    uploaded_files = st.file_uploader(
        "Upload documents to chat with",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=not _configured,
    )

    # Ingest button — only shown when files are selected
    if uploaded_files and st.button("Ingest Documents", disabled=not _configured):
        # Lazy imports to avoid loading ingestion/vectorstore modules until needed
        from ingestion.ingest import SPLITTER, SUPPORTED_EXTENSIONS, load_document
        from rag.vectorstore import get_vector_store

        with st.spinner("Ingesting documents..."):
            vector_store = get_vector_store()

            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                if suffix.lower() not in SUPPORTED_EXTENSIONS:
                    st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                    continue

                # Streamlit's uploaded_file is an in-memory file object.
                # LangChain's document loaders need a file PATH on disk.
                # So we write to a temp file, process it, then clean up.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    # Load the document into LangChain Document objects
                    docs = load_document(tmp_path)
                    # Split into chunks (1000 chars with 200 char overlap)
                    chunks = SPLITTER.split_documents(docs)
                    # Index chunks into Azure AI Search (embeds + stores them)
                    vector_store.add_documents(chunks)
                    st.success(
                        f"Ingested {uploaded_file.name} ({len(chunks)} chunks)"
                    )
                finally:
                    # Always clean up the temp file, even if ingestion fails
                    tmp_path.unlink(missing_ok=True)

    st.divider()

    # Clear chat history button — wipes conversation memory for this session
    if st.button("Clear Chat History"):
        from rag.chain import clear_session_history

        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        # st.rerun() forces Streamlit to re-execute the script immediately,
        # which re-renders the UI with the now-empty message list.
        st.rerun()

# ─── Chat Display ─────────────────────────────────────────────────────────────
# Re-render all previous messages from session_state on each rerun.
# This is how Streamlit maintains the chat history visually — the actual
# data lives in st.session_state.messages, and we re-draw it each time.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this message has source citations, show them in a collapsible section
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['name']}**")
                    st.caption(src["snippet"])

# ─── Chat Input ───────────────────────────────────────────────────────────────
# st.chat_input returns the user's message when they press Enter, or None.
# The walrus operator (:=) assigns the value AND checks if it's truthy.
if not _configured:
    # Show a disabled input box when credentials aren't configured
    st.chat_input("Ask a question about your documents", disabled=True)
elif prompt := st.chat_input("Ask a question about your documents"):
    # 1. Save the user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the RAG chain and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the RAG chain with the user's question.
            # session_id is passed via config so RunnableWithMessageHistory
            # knows which conversation history to load/save.
            result = st.session_state.chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        # Display the answer text
        st.markdown(answer)

        # Display source citations in a collapsible section.
        # This lets users verify the answer by seeing which document chunks
        # were retrieved and used as context for the LLM.
        sources = []
        if source_documents:
            with st.expander(f"Sources ({len(source_documents)})"):
                for i, doc in enumerate(source_documents, 1):
                    # Try to get the source file path from metadata, fall back to
                    # title, then a generic "Chunk N" label
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

    # 3. Save the assistant's response to session state for re-rendering
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
