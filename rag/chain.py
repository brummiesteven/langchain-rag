"""Core RAG chain with conversational memory using LangGraph.

RAG = Retrieval Augmented Generation. Instead of the LLM answering from its
training data alone, we first RETRIEVE relevant documents from a vector store,
then GENERATE an answer grounded in those documents. This dramatically reduces
hallucination and lets the LLM answer questions about YOUR specific documents.

THE PIPELINE (5 stages, expressed as a LangGraph StateGraph):

  User asks: "What about Q3?"
       │
       ▼
  ┌─ Stage 1: SUMMARIZE ──────────────────────────────────────────────────┐
  │ If conversation history exceeds _MAX_HISTORY_TOKENS, summarize       │
  │ older messages and remove them from state. The summary is stored     │
  │ in the `summary` field and injected into prompts at read-time.       │
  │ Keeps the last _KEEP_MESSAGES messages verbatim for context.         │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 2: CONDENSE ───────────────────────────────────────────────────┐
  │ If there's chat history, rewrite the follow-up question into a       │
  │ standalone question. "What about Q3?" → "What does the financial     │
  │ report say about Q3 performance?"                                    │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 3: RETRIEVE ───────────────────────────────────────────────────┐
  │ The standalone question is converted to a vector and used to search  │
  │ Azure AI Search for the top 4 most similar document chunks.          │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 4: FORMAT CONTEXT ─────────────────────────────────────────────┐
  │ Join the retrieved document chunks into a single string separated    │
  │ by double newlines. This string gets inserted into the prompt.       │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 5: GENERATE ANSWER ────────────────────────────────────────────┐
  │ The RAG prompt (with context + question + history) is sent to the    │
  │ LLM. The AI response is appended to messages for history tracking.   │
  └────────────────────────────────────────────────────────────────────────┘

CHAT HISTORY:
  LangGraph's InMemorySaver checkpointer persists conversation state per
  thread_id. Each Streamlit browser tab gets a unique thread_id (UUID).
  The summarize node compresses old messages when history exceeds a token
  threshold — functionally equivalent to SummarizationMiddleware but
  implemented as a graph node (the native pattern for custom StateGraphs).

  The summary is stored ONLY in the `summary` state field (single source of
  truth) and injected into prompts via _build_chat_history() at read-time.
  This avoids SystemMessage accumulation in the messages list.

  History is stored in-memory and lost on restart — fine for a single-process
  app. For production, swap InMemorySaver for PostgresSaver.
"""

from __future__ import annotations

import logging
import operator

from typing import Annotated  # noqa: F401 — required for get_type_hints() on Python 3.9

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,  # noqa: F401 — required for get_type_hints() on Python 3.9
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages  # noqa: F401 — required for get_type_hints() on Python 3.9

# ─── Logging ─────────────────────────────────────────────────────────────────
# Python's logging module lets us emit structured messages at different severity
# levels (DEBUG, INFO, WARNING, ERROR). getLogger(__name__) creates a logger
# named after this module (e.g. "rag.chain"), so log output can be filtered
# per-module. By default, only WARNING+ is shown; set logging.basicConfig(level=
# logging.DEBUG) in your entry point to see everything.
logger = logging.getLogger(__name__)

# ─── Local Imports ───────────────────────────────────────────────────────────
# These import from other modules in the rag package. They're separated from
# third-party imports above to follow Python convention (stdlib → third-party →
# local). Each provides a factory function used to build the RAG pipeline.
from rag.llm import get_llm
from rag.prompts import CONDENSE_PROMPT, RAG_PROMPT
from rag.vectorstore import get_retriever

# ─── Summarization Settings ──────────────────────────────────────────────────
# When conversation history exceeds _MAX_HISTORY_TOKENS, older messages are
# summarized into a concise text stored in the `summary` state field. The most
# recent _KEEP_MESSAGES are kept verbatim for immediate conversational context.
# 3 Q&A turns = 6 messages, which typically exceed 500 tokens. So on the 4th
# question, the oldest messages are summarized and the last 2 Q&A pairs (4
# messages) are kept verbatim. This keeps context tight and costs low.
_MAX_HISTORY_TOKENS = 500
_KEEP_MESSAGES = 4


# ─── State Schema ─────────────────────────────────────────────────────────────


class RAGState(MessagesState):
    """Graph state extending MessagesState with RAG pipeline fields.

    MessagesState provides `messages` with an add-reducer that automatically
    appends new messages and handles RemoveMessage for deletion. The
    checkpointer persists the full state (including messages) per thread_id.
    """

    standalone_question: str
    source_documents: list[Document]
    context: str
    answer: str
    summary: str
    # Flag set by summarize_node — True when old messages were actually
    # compressed in this invocation. The frontend reads this to show a
    # "history compacted" indicator so the user knows what happened.
    summarized: bool
    # Accumulates LLM prompt/response pairs from each pipeline node.
    # operator.add tells LangGraph to concatenate lists from each node
    # rather than overwriting, so {"llm_io": [item]} from each node stacks up.
    llm_io: Annotated[list, operator.add]


# ─── Utility Functions ────────────────────────────────────────────────────────


def format_docs(docs: list[Document]) -> str:
    """Join document contents with double newlines for use as LLM context.

    Takes a list of Document objects (each has .page_content with the text)
    and joins them into a single string. This string is what the LLM sees
    as "context" in the RAG prompt.

    Example:
        [Document("Chunk 1"), Document("Chunk 2")]
        → "Chunk 1\\n\\nChunk 2"
    """
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chat_history(messages: list, summary: str = "") -> list:
    """Build chat_history for prompt templates from state.

    Constructs the message list that gets injected into CONDENSE_PROMPT and
    RAG_PROMPT via MessagesPlaceholder("chat_history"). Excludes the last
    message (current user input) and optionally prepends the running
    conversation summary as a SystemMessage.

    The summary is stored in the `summary` state field (NOT as a message in
    the messages list) to avoid accumulation. It is only materialised here,
    at read-time, when building prompt inputs.

    Args:
        messages: Full message list from state (including current user message).
        summary: Running conversation summary text, if any.

    Returns:
        List of messages suitable for prompt chat_history.
    """
    if not messages:
        return []
    history = list(messages[:-1])
    if summary:
        summary_msg = SystemMessage(
            content=f"Summary of earlier conversation: {summary}"
        )
        return [summary_msg] + history
    return history


def _format_messages(msgs) -> str:
    """Format a list of LangChain messages into a readable multi-line string.

    Each message is prefixed with its role label (System/Human/AI) so the
    rendered prompt is easy to read in the frontend expander.
    """
    lines = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, HumanMessage):
            role = "Human"
        elif isinstance(m, AIMessage):
            role = "AI"
        else:
            role = type(m).__name__
        lines.append(f"{role}: {m.content}")
    return "\n\n".join(lines)


# ─── Module-level Checkpointer ────────────────────────────────────────────────
# Shared across all threads. Each thread_id gets its own conversation state.
# NOT persistent — all history is lost when the Python process restarts.
# For production, swap for PostgresSaver or another durable backend.
_checkpointer = InMemorySaver()


def get_rag_chain():
    """Build the full RAG graph with checkpointer and conversation summarization.

    This is the main entry point for the app. It assembles the entire pipeline
    as a LangGraph StateGraph and returns a compiled graph that can be invoked:
        graph.invoke(
            {"messages": [HumanMessage(content="...")]},
            config={"configurable": {"thread_id": "some-uuid"}}
        )

    The question is derived from the last message in state — no need to pass
    it separately. Returns a compiled StateGraph. The result dict contains:
      - "answer": the generated response string
      - "source_documents": list of retrieved Document objects (for citations)
    """
    llm = get_llm()
    retriever = get_retriever()

    # ─── Node Functions ───────────────────────────────────────────────────

    def summarize_node(state: RAGState) -> dict:
        """Compress old messages when history exceeds token threshold.

        Runs before every pipeline execution. If the conversation history
        (excluding the current user message) exceeds _MAX_HISTORY_TOKENS AND
        there are more than _KEEP_MESSAGES messages, older messages are removed
        from state and replaced with a concise summary stored in the `summary`
        field. The summary is NOT added as a SystemMessage to messages — it is
        injected into prompts at read-time via _build_chat_history().

        On failure (e.g. transient LLM error), summarization is skipped
        gracefully and the user's question is still answered.
        """
        messages = state["messages"]
        # Only look at history before the current user message
        history = messages[:-1]

        if not history or len(history) <= _KEEP_MESSAGES:
            return {"summarized": False, "llm_io": []}

        token_count = count_tokens_approximately(history)
        if token_count <= _MAX_HISTORY_TOKENS:
            return {"summarized": False, "llm_io": []}

        logger.info("Summarization triggered: %d messages, ~%d tokens in history",
                     len(history), token_count)
        logger.debug("Keeping last %d messages, summarizing %d older messages",
                      _KEEP_MESSAGES, len(history) - _KEEP_MESSAGES)

        to_summarize = history[:-_KEEP_MESSAGES]

        # Build the summarization prompt, incorporating any existing summary
        existing_summary = state.get("summary", "")
        if existing_summary:
            prompt = (
                f"This is an existing summary of the conversation:\n"
                f"{existing_summary}\n\n"
                "Extend the summary by incorporating the following new messages:\n\n"
            )
        else:
            prompt = (
                "Create a concise summary of the following conversation:\n\n"
            )

        for msg in to_summarize:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = "System"
            prompt += f"{role}: {msg.content}\n"

        prompt += "\nProvide a concise summary preserving key facts and context:"

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = response.content
        except Exception:
            logger.warning("Summarization failed; skipping and continuing pipeline")
            return {"summarized": False, "llm_io": []}

        # Remove old messages via RemoveMessage (processed by the add-reducer).
        # Summary is stored in state only — NOT as a SystemMessage in messages.
        delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize]

        return {
            "summary": summary,
            "messages": delete_msgs,
            "summarized": True,
            "llm_io": [{
                "label": "Summarize History",
                "prompt": prompt,
                "response": summary,
            }],
        }

    def condense_node(state: RAGState) -> dict:
        """Rewrite follow-ups into standalone questions; pass-through if no history.

        The question is derived from the last message in state (always a
        HumanMessage). If there's no prior chat history, we skip the condense
        step entirely to avoid an unnecessary LLM call on the first turn.
        """
        question = state["messages"][-1].content
        history = _build_chat_history(
            state["messages"], state.get("summary", "")
        )

        if history:
            logger.info("Condensing follow-up question with %d history messages", len(history))
            rendered = CONDENSE_PROMPT.format_messages(
                chat_history=history, question=question
            )
            response = llm.invoke(rendered)
            standalone = response.content
            logger.debug("Standalone question: %s", standalone)
            return {
                "standalone_question": standalone,
                "llm_io": [{
                    "label": "Condense Question",
                    "prompt": _format_messages(rendered),
                    "response": standalone,
                }],
            }
        logger.info("First turn — skipping condense step")
        return {"standalone_question": question, "llm_io": []}

    def retrieve_node(state: RAGState) -> dict:
        """Retrieve relevant documents using the standalone question."""
        query = state["standalone_question"]
        docs = retriever.invoke(query)
        logger.info("Retrieved %d documents for query: %s", len(docs), query)
        return {"source_documents": docs}

    def format_context_node(state: RAGState) -> dict:
        """Format retrieved documents into a context string."""
        return {"context": format_docs(state["source_documents"])}

    def generate_answer_node(state: RAGState) -> dict:
        """Generate the final answer and append AI response to messages.

        The answer chain receives the formatted document context, chat history
        (with summary prepended if present), and standalone question. The AI
        response is appended to messages so the checkpointer persists it for
        future turns.
        """
        logger.info("Generating answer")
        history = _build_chat_history(
            state["messages"], state.get("summary", "")
        )
        rendered = RAG_PROMPT.format_messages(
            context=state["context"],
            chat_history=history,
            question=state["standalone_question"],
        )
        response = llm.invoke(rendered)
        answer = response.content
        logger.debug("Answer length: %d characters", len(answer))
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "llm_io": [{
                "label": "Generate Answer",
                "prompt": _format_messages(rendered),
                "response": answer,
            }],
        }

    # ─── Build the Graph ──────────────────────────────────────────────────
    # LangGraph models pipelines as directed graphs. You define NODES (functions
    # that process state) and EDGES (the order they execute in). StateGraph
    # manages a shared state dict (RAGState) that flows through each node —
    # each node reads what it needs from state and returns updates to merge back.

    builder = StateGraph(RAGState)

    # Register each node function with a string name.
    # The name is used when defining edges and appears in debug traces.
    builder.add_node("summarize", summarize_node)
    builder.add_node("condense", condense_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("format_context", format_context_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Wire nodes into a linear pipeline: START → summarize → condense →
    # retrieve → format_context → generate_answer → END.
    # START and END are special sentinel nodes provided by LangGraph.
    # For branching/conditional flows, you'd use add_conditional_edges() instead.
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "condense")
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "format_context")
    builder.add_edge("format_context", "generate_answer")
    builder.add_edge("generate_answer", END)

    # compile() freezes the graph structure and attaches the checkpointer.
    # The checkpointer persists state between invocations (conversation memory).
    return builder.compile(checkpointer=_checkpointer)


def clear_session_history(graph, thread_id: str) -> None:
    """Clear all messages and summary for a given thread.

    Uses graph.get_state / graph.update_state to remove all messages via
    RemoveMessage, which updates the checkpointed state. This is the LangGraph
    equivalent of clearing the in-memory session history dict.
    """
    if graph is None:
        return

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
    except (ValueError, KeyError):
        return  # Thread doesn't exist — nothing to clear

    if not state.values or not state.values.get("messages"):
        return

    delete_messages = [RemoveMessage(id=m.id) for m in state.values["messages"]]
    if delete_messages:
        graph.update_state(config, {"messages": delete_messages, "summary": ""})
