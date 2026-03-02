"""Tests for the RAG chain utilities — document formatting and session history.

These tests cover the UTILITY FUNCTIONS in rag/chain.py, not the full RAG
pipeline (which would need mocked LLM and retriever clients). Specifically:

1. format_docs() — the function that joins Document objects into a context string
2. _get_session_history() — session history creation and retrieval
3. clear_session_history() — clearing conversation memory

These functions are simple, pure logic — no Azure calls needed — so they can
run without any mocking or credentials.
"""

from langchain_core.documents import Document

from rag.chain import _get_session_history, clear_session_history, format_docs


# ─── format_docs tests ────────────────────────────────────────────────────────
# format_docs takes a list of Document objects and joins their page_content
# with double newlines (\n\n). This is what gets injected as {context} in
# the RAG prompt.

def test_format_docs_single():
    """A single document should return its content as-is (no separators)."""
    docs = [Document(page_content="Hello world")]
    assert format_docs(docs) == "Hello world"


def test_format_docs_multiple():
    """Multiple documents should be separated by double newlines."""
    docs = [
        Document(page_content="First"),
        Document(page_content="Second"),
        Document(page_content="Third"),
    ]
    result = format_docs(docs)
    assert result == "First\n\nSecond\n\nThird"


def test_format_docs_empty():
    """An empty list should produce an empty string (no error)."""
    assert format_docs([]) == ""


# ─── Session history tests ────────────────────────────────────────────────────
# Each Streamlit session gets a unique session_id (UUID). The history store
# maps session_id → InMemoryChatMessageHistory. These tests verify the
# creation, retrieval, and clearing of session histories.

def test_session_history_created_on_access():
    """Accessing a new session_id should create a fresh empty history."""
    sid = "test-session-abc"
    history = _get_session_history(sid)
    # Should exist and be empty (no messages yet)
    assert history is not None
    assert len(history.messages) == 0
    # Accessing the same session_id again should return the SAME instance
    # (not create a new one), so messages accumulate correctly.
    assert _get_session_history(sid) is history
    # Cleanup so this test doesn't pollute other tests
    clear_session_history(sid)


def test_clear_session_history():
    """Clearing a session should remove all its messages."""
    sid = "test-session-clear"
    history = _get_session_history(sid)
    # Add a message to verify it gets cleared
    history.add_user_message("Hello")
    assert len(history.messages) == 1
    # Clear and verify
    clear_session_history(sid)
    assert len(_get_session_history(sid).messages) == 0


def test_clear_nonexistent_session():
    """Clearing a session that doesn't exist should not raise an error.

    This can happen if the user clicks "Clear Chat" before sending any messages,
    or if the session was already cleared. It should be a safe no-op.
    """
    clear_session_history("nonexistent-session-id")
