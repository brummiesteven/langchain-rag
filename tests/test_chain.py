"""Tests for the RAG chain utilities — document formatting and session history."""

from langchain_core.documents import Document

from rag.chain import _get_session_history, clear_session_history, format_docs


def test_format_docs_single():
    docs = [Document(page_content="Hello world")]
    assert format_docs(docs) == "Hello world"


def test_format_docs_multiple():
    docs = [
        Document(page_content="First"),
        Document(page_content="Second"),
        Document(page_content="Third"),
    ]
    result = format_docs(docs)
    assert result == "First\n\nSecond\n\nThird"


def test_format_docs_empty():
    assert format_docs([]) == ""


def test_session_history_created_on_access():
    sid = "test-session-abc"
    history = _get_session_history(sid)
    assert history is not None
    assert len(history.messages) == 0
    # Same session returns same instance
    assert _get_session_history(sid) is history
    # Cleanup
    clear_session_history(sid)


def test_clear_session_history():
    sid = "test-session-clear"
    history = _get_session_history(sid)
    history.add_user_message("Hello")
    assert len(history.messages) == 1
    clear_session_history(sid)
    assert len(_get_session_history(sid).messages) == 0


def test_clear_nonexistent_session():
    # Should not raise
    clear_session_history("nonexistent-session-id")
