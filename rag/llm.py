"""Factory functions for Azure OpenAI LLM and embedding clients.

Both clients route through Azure API Management (APIM), so every request
includes a Bearer token obtained via OAuth2 client credentials flow.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config import get_settings


def get_llm() -> AzureChatOpenAI:
    """Create an Azure OpenAI chat model instance via APIM."""
    settings = get_settings()
    token = settings.credential.get_token(settings.apim_scope).token
    return AzureChatOpenAI(
        azure_endpoint=settings.apim_endpoint,
        api_key="placeholder",
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.azure_openai_api_version,
        default_headers={
            "Authorization": f"Bearer {token}",
        },
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings instance via APIM."""
    settings = get_settings()
    token = settings.credential.get_token(settings.apim_scope).token
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.apim_endpoint,
        api_key="placeholder",
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version,
        default_headers={
            "Authorization": f"Bearer {token}",
        },
    )
