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
import time
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

if "history_summarized" not in st.session_state:
    # Tracks whether the conversation history has been compacted at least once.
    # Once True, the sidebar shows a persistent indicator so the user knows
    # older messages have been condensed into a summary. Reset on "Clear Chat".
    st.session_state.history_summarized = False

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

        clear_session_history(st.session_state.chain, st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.history_summarized = False
        # st.rerun() forces Streamlit to re-execute the script immediately,
        # which re-renders the UI with the now-empty message list.
        st.rerun()

    # ─── Conversation Status ──────────────────────────────────────────────
    # Persistent indicator that survives Streamlit reruns because it reads
    # from session_state (not from a one-shot result dict). Once the
    # summarize_node compresses old messages, this stays visible until the
    # user clears the chat.
    if st.session_state.history_summarized:
        st.divider()
        st.info(
            "Chat history has been compacted. "
            "Older messages were summarized to stay within token limits."
        )

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
        # If this message has LLM call data, show prompts and responses
        if msg.get("llm_io"):
            with st.expander(f"LLM Calls ({len(msg['llm_io'])})"):
                for call in msg["llm_io"]:
                    st.markdown(f"**{call['label']}**")
                    st.code(call["prompt"], language=None)
                    st.markdown("**Response:**")
                    st.code(call["response"], language=None)

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
            # Invoke the RAG graph with the user's question.
            # The HumanMessage is appended to the checkpointed messages
            # via the add-reducer. thread_id keys the conversation state.
            # The question is derived from messages[-1].content inside the
            # graph — no need to pass it separately.
            from langchain_community.callbacks import get_openai_callback
            from langchain_core.messages import HumanMessage

            # ── Response timing & token tracking ──────────────────────
            # time.perf_counter() is a high-resolution monotonic clock —
            # best choice for measuring elapsed wall-clock time.
            # get_openai_callback() is a LangChain context manager that
            # hooks into every OpenAI API call made inside the `with`
            # block and accumulates token counts (prompt, completion,
            # total). This captures ALL LLM calls in the pipeline:
            # condense, summarize, and answer generation.
            start_time = time.perf_counter()
            with get_openai_callback() as cb:
                result = st.session_state.chain.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.session_id,
                        }
                    },
                )
            elapsed = time.perf_counter() - start_time
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        # If the summarize_node compressed old messages on this turn,
        # flip the session flag so the sidebar indicator persists across reruns.
        if result.get("summarized"):
            st.session_state.history_summarized = True

        # Display the answer text
        st.markdown(answer)

        # Display response stats below the answer as small grey text.
        # st.caption() renders in a smaller, muted font — subtle but visible.
        # Token counts come from the OpenAI callback (cb) which tracked all
        # API calls during the invoke. The :, format adds thousand separators
        # for readability (e.g. 1,234 instead of 1234).
        st.caption(
            f"Response time: {elapsed:.1f}s | "
            f"Tokens: {cb.total_tokens:,} "
            f"(prompt: {cb.prompt_tokens:,}, completion: {cb.completion_tokens:,})"
        )

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

        # Display LLM prompts and responses in a collapsible section.
        # This shows the rendered prompt sent to the LLM and its raw response
        # for each pipeline stage (condense, summarize, answer generation).
        llm_io = result.get("llm_io", [])
        if llm_io:
            with st.expander(f"LLM Calls ({len(llm_io)})"):
                for call in llm_io:
                    st.markdown(f"**{call['label']}**")
                    st.code(call["prompt"], language=None)
                    st.markdown("**Response:**")
                    st.code(call["response"], language=None)

    # 3. Save the assistant's response to session state for re-rendering
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "llm_io": llm_io}
    )
