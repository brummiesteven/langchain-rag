"""Factory functions for Azure OpenAI LLM and embedding clients.

This module creates the two AI clients the app needs:
  1. AzureChatOpenAI  — for generating chat responses (the "brain" of RAG)
  2. AzureOpenAIEmbeddings — for converting text into vector embeddings
     (used to find similar documents in Azure AI Search)

IMPORTANT: Both clients route through Azure API Management (APIM), NOT directly
to Azure OpenAI. APIM is a gateway/proxy that sits in front of Azure OpenAI
and handles rate limiting, logging, and access control.

AUTHENTICATION: Instead of a static subscription key, we use OAuth2 client
credentials flow. On each call, we request a fresh Bearer token from Azure AD
using the ClientSecretCredential stored in Settings. This token is sent in the
Authorization header. APIM validates the token before forwarding the request to
Azure OpenAI.

TOKEN LIFECYCLE: credential.get_token() automatically caches tokens and refreshes
them when they expire (~1 hour). Since the LLM/embedding objects are typically
created once per Streamlit session (cached in st.session_state), the token is
set at creation time. For a PoC this is fine — sessions are usually much shorter
than an hour. For production, you'd want to rebuild the client periodically.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config import get_settings


def get_llm() -> AzureChatOpenAI:
    """Create an Azure OpenAI chat model instance that routes through APIM.

    Returns an AzureChatOpenAI object configured to:
    - Send requests to the APIM gateway endpoint (not Azure OpenAI directly)
    - Authenticate via a Bearer token in the Authorization header
    - Use the specified chat deployment (e.g. "gpt-4o")

    The api_key="placeholder" is required because LangChain's Azure OpenAI
    client validates that an API key is present (it throws an error if empty).
    However, APIM doesn't use this key — it validates the Bearer token instead.
    The placeholder value is never sent as an auth header because we override
    authentication with the Authorization header in default_headers.
    """
    settings = get_settings()

    # Request an OAuth2 access token for the APIM scope.
    # credential.get_token() returns an AccessToken object with .token (the JWT string)
    # and .expires_on (unix timestamp). The SDK handles caching and refresh internally.
    token = settings.credential.get_token(settings.apim_scope).token

    return AzureChatOpenAI(
        # The APIM gateway URL — all requests go here first, then APIM forwards
        # them to the underlying Azure OpenAI resource.
        azure_endpoint=settings.apim_endpoint,

        # LangChain requires a non-empty api_key for validation purposes.
        # This value is NOT used for actual auth — the Bearer token handles that.
        api_key="placeholder",

        # Which model deployment to use (configured in Azure OpenAI Studio).
        # This gets appended to the URL path, e.g. /openai/deployments/gpt-4o/chat/completions
        azure_deployment=settings.azure_openai_chat_deployment,

        # The Azure OpenAI REST API version. Different versions support different
        # features. We default to 2024-06-01 which supports GPT-4o.
        api_version=settings.azure_openai_api_version,

        # These headers are sent on EVERY HTTP request made by this client.
        # The Authorization header carries the OAuth2 Bearer token that APIM
        # validates before forwarding the request to Azure OpenAI.
        default_headers={
            "Authorization": f"Bearer {token}",
        },
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings instance that routes through APIM.

    Embeddings convert text into dense vector representations (arrays of floats).
    These vectors capture semantic meaning — similar texts produce similar vectors.
    We use embeddings to:
    - Index document chunks into Azure AI Search (during ingestion)
    - Convert user queries into vectors for similarity search (during retrieval)

    The configuration pattern is identical to get_llm() above — same APIM gateway,
    same Bearer token auth, just a different deployment name (embedding model
    instead of chat model).
    """
    settings = get_settings()

    # Get a fresh Bearer token (or cached if still valid) — same as get_llm()
    token = settings.credential.get_token(settings.apim_scope).token

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.apim_endpoint,
        api_key="placeholder",
        # Points to the embedding model deployment (e.g. "text-embedding-ada-002")
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version,
        default_headers={
            "Authorization": f"Bearer {token}",
        },
    )
