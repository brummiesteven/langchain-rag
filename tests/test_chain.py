"""Tests for the RAG chain — document formatting, chat history, session
management, and full pipeline integration.

Coverage:
1. format_docs()         — joins Document objects into a context string
2. _build_chat_history() — constructs prompt chat_history with optional summary
3. clear_session_history()— clears checkpointer state for a thread
4. Graph compilation     — smoke test that the graph builds correctly
5. Graph end-to-end      — full pipeline with mocked LLM and retriever
6. Summarization         — verifies old messages are pruned and summary is stored
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from rag.chain import _build_chat_history, clear_session_history, format_docs


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


# ─── _build_chat_history tests ────────────────────────────────────────────────
# _build_chat_history constructs the list injected into prompts via
# MessagesPlaceholder("chat_history"). It excludes the last message (current
# user input) and optionally prepends the summary as a SystemMessage.

def test_build_chat_history_no_summary():
    """Without a summary, returns all messages except the last."""
    messages = [
        HumanMessage(content="Q1"),
        AIMessage(content="A1"),
        HumanMessage(content="Q2"),
    ]
    history = _build_chat_history(messages)
    assert len(history) == 2
    assert history[0].content == "Q1"
    assert history[1].content == "A1"


def test_build_chat_history_with_summary():
    """With a summary, prepends a SystemMessage containing it."""
    messages = [
        HumanMessage(content="Q1"),
        AIMessage(content="A1"),
        HumanMessage(content="Q2"),
    ]
    history = _build_chat_history(messages, summary="User asked about Q1")
    assert len(history) == 3
    assert isinstance(history[0], SystemMessage)
    assert "User asked about Q1" in history[0].content
    # Actual conversation messages follow the summary
    assert history[1].content == "Q1"
    assert history[2].content == "A1"


def test_build_chat_history_first_message():
    """First message (no prior history) should return empty list."""
    messages = [HumanMessage(content="Q1")]
    history = _build_chat_history(messages)
    assert history == []


def test_build_chat_history_first_message_with_summary():
    """First message with a leftover summary returns just the summary."""
    messages = [HumanMessage(content="Q1")]
    history = _build_chat_history(messages, summary="old summary")
    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)


def test_build_chat_history_empty_messages():
    """Empty messages list should return empty list."""
    assert _build_chat_history([]) == []
    assert _build_chat_history([], summary="anything") == []


# ─── Session history tests ────────────────────────────────────────────────────
# clear_session_history works with the LangGraph checkpointer via
# graph.get_state / graph.update_state.

def test_clear_session_history():
    """Clearing a session should remove all messages and reset summary."""
    mock_graph = MagicMock()
    mock_state = MagicMock()
    mock_state.values = {
        "messages": [
            HumanMessage(content="Hello", id="msg-1"),
            AIMessage(content="Hi there", id="msg-2"),
        ],
    }
    mock_graph.get_state.return_value = mock_state

    clear_session_history(mock_graph, "test-thread")

    # Verify get_state was called with correct config
    mock_graph.get_state.assert_called_once_with(
        {"configurable": {"thread_id": "test-thread"}}
    )
    # Verify update_state was called with RemoveMessages + empty summary (T4)
    mock_graph.update_state.assert_called_once()
    call_args = mock_graph.update_state.call_args[0]
    state_update = call_args[1]
    assert state_update["summary"] == ""
    assert len(state_update["messages"]) == 2
    assert all(isinstance(m, RemoveMessage) for m in state_update["messages"])


def test_clear_nonexistent_session():
    """Clearing a session that doesn't exist should not raise an error.

    This can happen if the user clicks "Clear Chat" before sending any messages,
    or if the session was already cleared. It should be a safe no-op.
    """
    mock_graph = MagicMock()
    mock_graph.get_state.side_effect = ValueError("Thread not found")

    # Should not raise
    clear_session_history(mock_graph, "nonexistent-session-id")
    mock_graph.update_state.assert_not_called()


def test_clear_empty_session():
    """Clearing a session with no messages should be a no-op."""
    mock_graph = MagicMock()
    mock_state = MagicMock()
    mock_state.values = {"messages": []}
    mock_graph.get_state.return_value = mock_state

    clear_session_history(mock_graph, "test-empty")
    mock_graph.update_state.assert_not_called()


def test_clear_with_none_graph():
    """Clearing when graph is None (unconfigured mode) should be a safe no-op."""
    clear_session_history(None, "any-session-id")


# ─── Graph compilation smoke test ─────────────────────────────────────────────

@patch("rag.chain.get_retriever")
@patch("rag.chain.get_llm")
def test_graph_compiles(mock_get_llm, mock_get_retriever):
    """The RAG graph should compile without error."""
    mock_get_llm.return_value = MagicMock()
    mock_get_retriever.return_value = MagicMock()

    from rag.chain import get_rag_chain

    graph = get_rag_chain()
    assert graph is not None


# ─── Graph end-to-end tests ──────────────────────────────────────────────────
# These tests invoke the compiled graph with mocked LLM and retriever,
# verifying that state flows correctly through all nodes.

def _make_mock_llm():
    """Create a mock LLM that works with both LCEL chains and direct calls."""
    mock_llm = MagicMock()
    # For LCEL chains: PROMPT | mock_llm | StrOutputParser()
    # The mock is wrapped in RunnableLambda, called directly → .return_value
    mock_llm.return_value = AIMessage(content="mocked answer")
    # For direct calls: llm.invoke([...]) in summarize_node
    mock_llm.invoke.return_value = AIMessage(content="mocked summary")
    return mock_llm


def _make_mock_retriever():
    """Create a mock retriever returning a single test document."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="relevant content", metadata={"source": "test.pdf"})
    ]
    return mock_retriever


@patch("rag.chain.get_retriever")
@patch("rag.chain.get_llm")
def test_graph_first_turn(mock_get_llm, mock_get_retriever):
    """First turn: no history → condense is skipped, answer is generated."""
    mock_get_llm.return_value = _make_mock_llm()
    mock_get_retriever.return_value = _make_mock_retriever()

    from rag.chain import get_rag_chain

    graph = get_rag_chain()
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is X?")]},
        config={"configurable": {"thread_id": "test-first-turn"}},
    )

    assert result["answer"] == "mocked answer"
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0].page_content == "relevant content"


@patch("rag.chain.get_retriever")
@patch("rag.chain.get_llm")
def test_graph_multi_turn_preserves_history(mock_get_llm, mock_get_retriever):
    """Multi-turn: messages accumulate in checkpointer across invocations."""
    mock_get_llm.return_value = _make_mock_llm()
    mock_get_retriever.return_value = _make_mock_retriever()

    from rag.chain import get_rag_chain

    graph = get_rag_chain()
    config = {"configurable": {"thread_id": "test-multi-turn"}}

    # Turn 1
    graph.invoke({"messages": [HumanMessage(content="Q1")]}, config=config)
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 2  # HumanMessage + AIMessage

    # Turn 2
    graph.invoke({"messages": [HumanMessage(content="Q2")]}, config=config)
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 4  # Q1, A1, Q2, A2


# ─── Summarization test ──────────────────────────────────────────────────────
# Patch thresholds low so summarization triggers after just a few turns.

@patch("rag.chain._KEEP_MESSAGES", 2)
@patch("rag.chain._MAX_HISTORY_TOKENS", 0)  # Always exceed threshold
@patch("rag.chain.get_retriever")
@patch("rag.chain.get_llm")
def test_summarization_prunes_old_messages(mock_get_llm, mock_get_retriever):
    """When history exceeds thresholds, old messages are pruned and summary stored."""
    mock_get_llm.return_value = _make_mock_llm()
    mock_get_retriever.return_value = _make_mock_retriever()

    from rag.chain import get_rag_chain

    graph = get_rag_chain()
    config = {"configurable": {"thread_id": "test-summarization"}}

    # Turns 1 and 2: build up history (4 messages after 2 turns)
    graph.invoke({"messages": [HumanMessage(content="Q1")]}, config=config)
    graph.invoke({"messages": [HumanMessage(content="Q2")]}, config=config)
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 4  # Q1, A1, Q2, A2

    # Turn 3: history has 4 messages, _KEEP_MESSAGES=2, tokens>0 → summarize
    # to_summarize = first 2 messages (Q1, A1), keep last 2 (Q2, A2)
    graph.invoke({"messages": [HumanMessage(content="Q3")]}, config=config)
    state = graph.get_state(config)

    # Summary should be populated
    assert state.values.get("summary") == "mocked summary"
    # Summarized flag should be set
    assert state.values.get("summarized") is True
    # Messages should be bounded: kept messages (Q2, A2) + current turn (Q3, A3)
    assert len(state.values["messages"]) == 4
    # First message should now be Q2 (Q1 was pruned)
    assert state.values["messages"][0].content == "Q2"
