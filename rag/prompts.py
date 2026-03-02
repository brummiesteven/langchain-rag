"""Prompt templates for the two-stage RAG pipeline.

WHAT ARE PROMPT TEMPLATES?
  Prompt templates are pre-written instructions for the LLM with placeholder
  variables (like {question} and {context}) that get filled in at runtime.
  They use ChatPromptTemplate which structures the prompt as a list of messages
  with roles (system, human) — this is how chat-based LLMs expect their input.

WHY TWO PROMPTS?
  The RAG pipeline has two LLM calls, each needing different instructions:

  1. CONDENSE_PROMPT (Stage 1) — Rewrites follow-up questions into standalone
     queries. Without this, a question like "Tell me more about that" would be
     sent to the retriever as-is and return irrelevant results. The condense
     step uses chat history to understand what "that" refers to and produces
     a self-contained question like "What are the details of the Q3 revenue
     decline mentioned in the financial report?"

  2. RAG_PROMPT (Stage 4) — The main answer generation prompt. It receives the
     retrieved document chunks as {context} and instructs the LLM to answer
     ONLY based on that context. This grounding is what prevents hallucination.

MESSAGES PLACEHOLDER:
  MessagesPlaceholder("chat_history") is a dynamic slot that gets replaced
  with the actual list of past messages at runtime. It's inserted between the
  system message and the human message, so the LLM sees:
    [system instruction] → [past messages...] → [current question]
  This gives the LLM full conversational context.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ─── Stage 1: Condense Prompt ────────────────────────────────────────────────
# Used to rewrite follow-up questions into standalone questions.
#
# Example transformation:
#   Chat history: "User: What was Q3 revenue? AI: Q3 revenue was $50M."
#   Follow-up: "How does that compare to Q2?"
#   Output: "How does the Q3 revenue of $50M compare to Q2 revenue?"
#
# The system message tells the LLM its job (rephrase the question).
# MessagesPlaceholder injects the conversation history.
# The human message contains the current follow-up question.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question. "
            "If the follow-up question is already standalone, return it as-is.",
        ),
        # This placeholder is replaced at runtime with the actual chat history
        # (a list of HumanMessage and AIMessage objects from previous turns)
        MessagesPlaceholder("chat_history"),
        # The {question} placeholder is filled with the user's current input
        ("human", "{question}"),
    ]
)

# ─── Stage 2: RAG Answer Prompt ──────────────────────────────────────────────
# Used to generate the final answer grounded in retrieved documents.
#
# The key instruction is "Answer based ONLY on the following context" —
# this grounds the LLM in the actual documents and prevents hallucination.
# If the retrieved context doesn't contain enough information, the LLM is
# instructed to say so rather than making something up.
#
# The {context} placeholder is filled with the formatted document chunks
# (joined with double newlines) from Stage 3 of the pipeline.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question based only "
            "on the following context. If the context does not contain enough "
            "information to answer, say so.\n\n"
            "Context:\n{context}",
        ),
        # Chat history provides multi-turn context so the LLM can reference
        # things discussed in previous turns
        MessagesPlaceholder("chat_history"),
        # The standalone question (after condense step) or original question
        ("human", "{question}"),
    ]
)
