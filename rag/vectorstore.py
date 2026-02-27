"""Azure AI Search vector store and retriever factories.

Connects directly to Azure AI Search (not via APIM) using the search
service endpoint and admin key.
"""

from langchain_community.vectorstores import AzureSearch
from langchain_core.vectorstores import VectorStoreRetriever

from config import get_settings
from rag.llm import get_embeddings


def get_vector_store() -> AzureSearch:
    """Create an AzureSearch vector store backed by Azure AI Search."""
    settings = get_settings()
    embeddings = get_embeddings()
    return AzureSearch(
        azure_search_endpoint=settings.azure_search_endpoint,
        azure_search_key=settings.azure_search_key,
        index_name=settings.azure_search_index_name,
        # AzureSearch expects a callable, not an Embeddings object
        embedding_function=embeddings.embed_query,
    )


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Create a retriever that returns the top-k most relevant documents."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
