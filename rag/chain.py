"""Core RAG chain with conversational memory.

Implements a two-stage LCEL pipeline:
  1. Condense — rewrites follow-up questions into standalone queries.
  2. Retrieve + Generate — fetches relevant docs from Azure AI Search,
     then generates an answer grounded in those docs.

Chat history is managed per-session using InMemoryChatMessageHistory
(lost on restart — fine for a single-process Streamlit app).
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.llm import get_llm
from rag.prompts import CONDENSE_PROMPT, RAG_PROMPT
from rag.vectorstore import get_retriever

# In-memory store of chat histories keyed by session ID.
# Not persistent — all history is lost when the process restarts.
_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the chat history for a session, creating one if needed."""
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def clear_session_history(session_id: str) -> None:
    """Clear all messages for a given session."""
    if session_id in _session_histories:
        _session_histories[session_id].clear()


def format_docs(docs: list[Document]) -> str:
    """Join document contents with double newlines for use as LLM context."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain() -> RunnableWithMessageHistory:
    """Build the full RAG chain with chat history support.

    Returns a dict with keys:
      - "answer": the generated response string
      - "source_documents": list of retrieved Document objects

    The chain is wrapped in RunnableWithMessageHistory so chat history
    is automatically loaded/saved per session_id.
    """
    llm = get_llm()
    retriever = get_retriever()

    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def condense_question(input: dict) -> str:
        """Rewrite follow-ups into standalone questions; pass-through if no history."""
        if input.get("chat_history"):
            return condense_chain.invoke(input)
        return input["question"]

    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    rag_chain = (
        # Stage 1: Condense follow-up into a standalone question
        RunnablePassthrough.assign(
            standalone_question=condense_question,
        )
        # Stage 2: Retrieve relevant documents from the vector store
        | RunnablePassthrough.assign(
            source_documents=lambda x: retriever.invoke(x["standalone_question"]),
        )
        # Stage 3: Format docs as context string for the prompt
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source_documents"]),
            question=lambda x: x["standalone_question"],
        )
        # Stage 4: Generate the answer via the RAG prompt + LLM
        | RunnablePassthrough.assign(
            answer=answer_chain,
        )
        # Stage 5: Return only the answer and sources (drop intermediate keys)
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "source_documents": x["source_documents"],
            }
        )
    )

    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="question",       # key in input dict containing the user query
        history_messages_key="chat_history",  # key the prompt template expects for history
        output_messages_key="answer",         # key in output dict to save as AI message
    )


def get_simple_rag_chain():
    """One-shot RAG chain without history — useful for CLI testing."""
    llm = get_llm()
    retriever = get_retriever()

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
