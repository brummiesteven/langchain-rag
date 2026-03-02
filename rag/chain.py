"""Core RAG chain with conversational memory.

RAG = Retrieval Augmented Generation. Instead of the LLM answering from its
training data alone, we first RETRIEVE relevant documents from a vector store,
then GENERATE an answer grounded in those documents. This dramatically reduces
hallucination and lets the LLM answer questions about YOUR specific documents.

THE PIPELINE (5 stages):

  User asks: "What about Q3?"
       │
       ▼
  ┌─ Stage 1: CONDENSE ─────────────────────────────────────────────────┐
  │ If there's chat history, rewrite the follow-up question into a     │
  │ standalone question. "What about Q3?" → "What does the financial   │
  │ report say about Q3 performance?" Without this, the retriever      │
  │ would search for "What about Q3?" and get irrelevant results.      │
  └─────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 2: RETRIEVE ─────────────────────────────────────────────────┐
  │ The standalone question is converted to a vector (embedding) and   │
  │ used to search Azure AI Search for the top 4 most similar document │
  │ chunks. Returns a list of Document objects with page_content.      │
  └─────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 3: FORMAT CONTEXT ───────────────────────────────────────────┐
  │ Join the retrieved document chunks into a single string separated  │
  │ by double newlines. This string gets inserted into the prompt      │
  │ template as the {context} variable.                                │
  └─────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 4: GENERATE ANSWER ──────────────────────────────────────────┐
  │ The RAG prompt (with context + question) is sent to the LLM.       │
  │ The LLM generates an answer based ONLY on the provided context.    │
  └─────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Stage 5: CLEAN OUTPUT ─────────────────────────────────────────────┐
  │ Strip intermediate keys (standalone_question, context) and return  │
  │ only {"answer": "...", "source_documents": [...]}.                 │
  └─────────────────────────────────────────────────────────────────────┘

LCEL (LangChain Expression Language):
  The chain is built using LCEL, which lets you compose steps with the | (pipe)
  operator. Data flows left to right, like Unix pipes. Each stage receives the
  output of the previous stage as its input.

CHAT HISTORY:
  RunnableWithMessageHistory wraps the chain and automatically loads/saves
  conversation history per session_id. This lets users have multi-turn
  conversations ("Tell me more about that" works because history provides context).
  History is stored in-memory and lost on restart — fine for a single-process app.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.llm import get_llm
from rag.prompts import CONDENSE_PROMPT, RAG_PROMPT
from rag.vectorstore import get_retriever

# ─── Session History Storage ──────────────────────────────────────────────────
# A simple in-memory dict mapping session_id -> InMemoryChatMessageHistory.
# Each Streamlit browser tab gets a unique session_id (a UUID).
# NOT persistent — all history is lost when the Python process restarts.
# For production, you'd swap this for Redis, Cosmos DB, or a database.
_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the chat history for a session, creating a new one if needed.

    This function is passed to RunnableWithMessageHistory. Before each chain
    invocation, LangChain calls this to get the history object for the current
    session, then injects the stored messages into the prompt template.
    """
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def clear_session_history(session_id: str) -> None:
    """Clear all messages for a given session (called by the UI's clear button)."""
    if session_id in _session_histories:
        _session_histories[session_id].clear()


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


def get_rag_chain() -> RunnableWithMessageHistory:
    """Build the full RAG chain with chat history support.

    This is the main entry point for the app. It assembles the entire pipeline
    and returns a chain that can be invoked with:
        chain.invoke(
            {"question": "user's question"},
            config={"configurable": {"session_id": "some-uuid"}}
        )

    Returns a dict with:
      - "answer": the generated response string
      - "source_documents": list of retrieved Document objects (for citations)
    """
    # Create the LLM (chat model) and retriever (vector search) instances.
    # These are created once when the chain is built. The chain itself is
    # cached in st.session_state, so this only runs once per Streamlit session.
    llm = get_llm()
    retriever = get_retriever()

    # Sub-chain for Stage 1: takes chat_history + question, outputs a
    # standalone question string. Uses LCEL pipe syntax:
    # prompt template → LLM → extract string from LLM response
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def condense_question(input: dict) -> str:
        """Rewrite follow-ups into standalone questions; pass-through if no history.

        If there's no chat history (first message in the conversation), we skip
        the condense step entirely and just use the question as-is. This avoids
        an unnecessary LLM call on the first turn.
        """
        if input.get("chat_history"):
            return condense_chain.invoke(input)
        return input["question"]

    # Sub-chain for Stage 4: takes context + question, outputs an answer string.
    # RAG_PROMPT contains the system instructions ("answer based only on context")
    # plus placeholders for {context}, {question}, and chat_history.
    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    # ─── The Main Pipeline ────────────────────────────────────────────────
    # Built using RunnablePassthrough.assign() which passes the input dict
    # through unchanged BUT adds a new key computed by the function/chain.
    # At each stage, the dict accumulates more keys.
    #
    # Input dict starts as: {"question": "user's question", "chat_history": [...]}
    rag_chain = (
        # Stage 1: Add "standalone_question" key to the dict.
        # If there's chat history, this rewrites the question to be self-contained.
        # After this: {..., "standalone_question": "rewritten question"}
        RunnablePassthrough.assign(
            standalone_question=condense_question,
        )
        # Stage 2: Add "source_documents" key — the retrieved document chunks.
        # Uses the standalone question (not the original) for better retrieval.
        # After this: {..., "source_documents": [Document, Document, ...]}
        | RunnablePassthrough.assign(
            source_documents=lambda x: retriever.invoke(x["standalone_question"]),
        )
        # Stage 3: Add "context" key (formatted docs string) and copy the
        # standalone question into "question" (overwriting the original).
        # The answer_chain's prompt template expects {context} and {question}.
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source_documents"]),
            question=lambda x: x["standalone_question"],
        )
        # Stage 4: Add "answer" key — the LLM's generated response.
        # answer_chain reads {context}, {question}, and {chat_history} from the dict.
        | RunnablePassthrough.assign(
            answer=answer_chain,
        )
        # Stage 5: Drop all intermediate keys (standalone_question, context, etc.)
        # and return only what the UI needs: the answer text and source documents.
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "source_documents": x["source_documents"],
            }
        )
    )

    # Wrap the chain with automatic chat history management.
    # RunnableWithMessageHistory does the following on each invoke():
    #   1. Calls _get_session_history(session_id) to get the history object
    #   2. Injects past messages into the chain input under "chat_history"
    #   3. After the chain runs, saves the user's question as a HumanMessage
    #      and the answer as an AIMessage to the history
    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="question",       # Which input key is the user's message
        history_messages_key="chat_history",  # Which key the prompt expects for history
        output_messages_key="answer",         # Which output key is the AI's response
    )


def get_simple_rag_chain():
    """One-shot RAG chain without history — useful for CLI testing.

    A simpler version that skips the condense step and chat history.
    Good for testing retrieval + generation without the complexity of
    multi-turn conversation management.

    Usage:
        chain = get_simple_rag_chain()
        answer = chain.invoke({"question": "What is X?"})
    """
    llm = get_llm()
    retriever = get_retriever()

    return (
        # Retrieve docs and format as context string in one step
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | RAG_PROMPT     # Fill in the prompt template with context + question
        | llm            # Send to the LLM
        | StrOutputParser()  # Extract the text from the LLM's response
    )
