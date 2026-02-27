"""Prompt templates for the two-stage RAG pipeline.

CONDENSE_PROMPT — rewrites follow-up questions into standalone queries so
    retrieval works correctly (e.g. "Tell me more" becomes a full question).
RAG_PROMPT — generates an answer grounded in the retrieved context.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Stage 1: Condense a follow-up question using chat history into a
# standalone question that can be sent to the retriever.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question. "
            "If the follow-up question is already standalone, return it as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# Stage 2: Answer the question using only the retrieved context.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question based only "
            "on the following context. If the context does not contain enough "
            "information to answer, say so.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
