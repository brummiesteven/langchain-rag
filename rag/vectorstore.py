"""Azure AI Search vector store and retriever factories.

WHAT IS A VECTOR STORE?
  A vector store is a database optimised for storing and searching vectors
  (arrays of numbers). When we ingest a document, we:
    1. Split it into chunks (e.g. 1000-character paragraphs)
    2. Convert each chunk into a vector using an embedding model
    3. Store the vector + original text in the vector store

  When a user asks a question, we:
    1. Convert the question into a vector using the same embedding model
    2. Search the vector store for the most similar vectors (nearest neighbours)
    3. Return the corresponding text chunks as "retrieved documents"

  Similar texts produce similar vectors, so "What is the company's Q3 revenue?"
  will match chunks that discuss Q3 financial performance — even if they don't
  use the exact same words.

WHY AZURE AI SEARCH?
  Azure AI Search (formerly Azure Cognitive Search) is a managed search service
  that supports both traditional keyword search AND vector search. We use it as
  our vector store because it integrates well with the Azure ecosystem.

IMPORTANT: Azure AI Search is accessed DIRECTLY — NOT through APIM. It has its
own endpoint and admin key. Only the LLM and embedding calls go through APIM.
"""

from langchain_community.vectorstores import AzureSearch
from langchain_core.vectorstores import VectorStoreRetriever

from config import get_settings
from rag.llm import get_embeddings


def get_vector_store() -> AzureSearch:
    """Create an AzureSearch vector store backed by Azure AI Search.

    This object can both READ from and WRITE to the search index.
    - Write: vector_store.add_documents(chunks) — used during ingestion
    - Read: vector_store.as_retriever() — used during RAG to find relevant docs

    IMPORTANT SUBTLETY: AzureSearch expects `embedding_function` to be a
    CALLABLE (a function), not an Embeddings object. That's why we pass
    `embeddings.embed_query` (the bound method) rather than `embeddings` itself.
    If you pass the object, you'll get a cryptic error about wrong argument types.
    """
    settings = get_settings()
    embeddings = get_embeddings()

    return AzureSearch(
        # The URL of the Azure AI Search service
        azure_search_endpoint=settings.azure_search_endpoint,
        # Admin key for full read/write access to the index
        azure_search_key=settings.azure_search_key,
        # Name of the search index where documents are stored
        index_name=settings.azure_search_index_name,
        # A CALLABLE that takes a string and returns a vector (list of floats).
        # embeddings.embed_query is a method: str → list[float]
        # This is used internally whenever AzureSearch needs to vectorise text.
        embedding_function=embeddings.embed_query,
    )


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Create a retriever that returns the top-k most relevant documents.

    A retriever is a simplified interface over the vector store that just does
    search. You call retriever.invoke("some question") and get back a list of
    the k most relevant Document objects.

    Args:
        k: Number of documents to retrieve per query. Default is 4.
           More docs = more context for the LLM but also more tokens used.
           4 is a common starting point for RAG applications.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
