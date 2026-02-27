"""Factory functions for Azure OpenAI LLM and embedding clients.

Both clients route through Azure API Management (APIM), so every request
includes the Ocp-Apim-Subscription-Key header alongside the OpenAI API key.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config import get_settings


def get_llm() -> AzureChatOpenAI:
    """Create an Azure OpenAI chat model instance via APIM."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.azure_openai_api_version,
        # APIM gateway requires its own subscription key header
        default_headers={
            "Ocp-Apim-Subscription-Key": settings.apim_subscription_key,
        },
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings instance via APIM."""
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version,
        default_headers={
            "Ocp-Apim-Subscription-Key": settings.apim_subscription_key,
        },
    )
