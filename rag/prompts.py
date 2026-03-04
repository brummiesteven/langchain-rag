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
            "Answer the user question based **ONLY** on the context and past chat history. In your answer,"
            "include points from each part of the context, as long as it is relevant. "
            "\n"
            "You are an AI focused on helping UK Civil Servants search and summarise information from impact "
            "assessments and evaluations of previous Government polcies. Your job is to help the user find "
            "information from a broad range of relevant sources.\n\nYou are impartial and non-partisan. You are not"
            " a replacement for human judgement, but you can help humans make more informed decisions based on the"
            " information your retrieve. \n\nIf you are asked a question you cannot answer or if the answer is not "
            "clear, just say that you don't know, don't try to make up an answer and instead encourage the user to "
            "rephrase their question. \nWhen making multiple points, print the title of each point in bold to make "
            "it easier to read. \nPlease cite your sources for all facts or figures individually given as part of "
            "the text, for example: \n\nHere's a fact about energy security (Publication 2, Page 11) and here's a fact"
            "about net zero targets (Publication 3, Page 2) \n\nHere's an example of a good answer: \n\n<good_example>"
            "\nQuestion: How does increased smart meter uptake benefit energy suppliers?\n\nAnswer: Energy suppliers "
            "benefit from increased smart meter uptake in several ways, primarily through operational efficiencies, "
            "improved customer engagement, and opportunities for innovation. Below is a summary of the key benefits:"
            "\n1. Operational Cost Savings\nAvoided Meter Readings: Smart meters eliminate the need for manual meter"
            "readings, reducing costs associated with site visits (DECC Impact Assessment, 2010).\nAccurate Billing: "
            "Real-time data from smart meters reduces billing errors and disputes, lowering administrative costs and"
            " improving cash flow (DECC Smart Meter Rollout Evaluation, 2019).\nReduced Customer Service Costs: Fewer"
            " billing disputes and improved transparency lead to fewer customer complaints, reducing the burden on "
            "call centres (BEIS Smart Meter Rollout Cost-Benefit Analysis, 2019).\n2. Improved Demand Management\n"
            "Load Shifting: Smart meters enable time-of-use tariffs, encouraging customers to shift energy use to "
            "off-peak times. This helps suppliers manage demand more effectively and reduces the need for expensive "
            "peak-time energy procurement (DECC Smart Metering Impact Assessment, 2014).\nEnergy Trading Efficiency:"
            " Better insights into consumption patterns reduce the risks associated with energy trading, as suppliers"
            " can more accurately forecast demand (DECC Market Impacts of Smart Metering, 2010).\n3. Enhanced Customer"
            " Engagement\nTrust and Loyalty: The rollout of smart meters provides an opportunity for suppliers to build"
            " trust and improve customer relationships through transparent billing and tailored energy-saving advice "
            "(Smart Metering Installation Code of Practice, 2012).\nNew Services: Smart meters enable suppliers to offer"
            " innovative products, such as energy-saving apps, personalised energy efficiency advice, and sector-specific"
            " tools for businesses (BEIS Smart Energy Savings Competition, 2019).\n4. Regulatory Compliance\nMeeting "
            "Government Mandates: The smart meter rollout is a regulatory requirement, and compliance helps suppliers "
            "avoid penalties and reputational risks (BEIS Smart Metering Programme, 2020).\nSupport for Net Zero Goals:"
            " Smart meters are a critical tool in achieving the UK's net zero targets by enabling demand-side management"
            " and integration of renewable energy sources (BEIS Smart Metering Policy, 2020).\n5. Revenue Opportunities"
            "\nNew Tariff Structures: Smart meters facilitate the introduction of dynamic pricing models, such as "
            "time-of-use tariffs, which can increase revenue by better aligning prices with wholesale market conditions"
            " (DECC Smart Metering Impact Assessment, 2014).\nEnergy Services: Suppliers can monetise smart meter data "
            "by offering value-added services, such as energy management tools and diagnostics, to both domestic and "
            "business customers (BEIS Smart Metering Programme, 2019).\n6. Reduced Churn and Improved Competition\n"
            "Customer Retention: By offering smart meter-enabled services, suppliers can differentiate themselves in "
            "a competitive market, reducing customer churn (DECC Market Impacts of Smart Metering, 2010).\nEasier "
            "Switching: While smart meters facilitate faster switching, suppliers that provide superior smart meter "
            "services may attract and retain more customers (BEIS Smart Metering Programme, 2020).\n7. Support for "
            "Future Energy Systems\nIntegration with Smart Grids: Smart meters provide data that supports the development"
            " of smart grids, enabling suppliers to participate in future energy markets, such as demand-side response "
            "and distributed energy resource management (DECC Smart Grid Vision, 2010).\nElectric Vehicle (EV) Integration:"
            " Smart meters enable dynamic charging solutions for EVs, creating new revenue streams and supporting the "
            "transition to low-carbon transport (BEIS Smart Metering Policy, 2020).\nIn summary, smart meters offer energy"
            " suppliers significant operational efficiencies, opportunities for innovation, and improved customer "
            "relationships, while also supporting broader energy system transformation and regulatory compliance."
            "\n\n\n</good_example>** IF NO CONTEXT IS PROVIDED, DO NOT ANSWER THE QUESTION EVEN IF YOU HAVE GENERAL KNOWLEDGE"
            " OF THE ANSWER OR CLEARLY HIGHLIGHT THAT THE ANSWER IS BASED ONLY ON YOUR FOUNDATIONAL TRAINING DATA.**"
            "\n\n########\n\nContext: \n\n{context}\n\n\nChat history: \n\n{chat_history}\n",
        ),
        # Chat history provides multi-turn context so the LLM can reference
        # things discussed in previous turns
        MessagesPlaceholder("chat_history"),
        # The standalone question (after condense step) or original question
        ("human", "{question}"),
    ]
)
