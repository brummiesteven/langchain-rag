"""Centralised application settings loaded from environment variables.

Reads Azure OpenAI, APIM, and Azure AI Search configuration from a .env file
(via python-dotenv) and exposes a cached Settings singleton via get_settings().
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(var: str) -> str:
    """Read an environment variable or raise if it is missing/empty."""
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable container for all Azure service configuration.

    Fields default to None so the class can be imported without triggering
    env-var validation. Actual values are resolved in __post_init__.
    """

    azure_openai_endpoint: str = field(default=None)
    azure_openai_api_key: str = field(default=None)
    azure_openai_chat_deployment: str = field(default=None)
    azure_openai_embedding_deployment: str = field(default=None)
    azure_openai_api_version: str = field(default=None)
    apim_subscription_key: str = field(default=None)
    azure_search_endpoint: str = field(default=None)
    azure_search_key: str = field(default=None)
    azure_search_index_name: str = field(default=None)

    def __post_init__(self):
        defaults = {
            "azure_openai_endpoint": _require("AZURE_OPENAI_ENDPOINT"),
            "azure_openai_api_key": _require("AZURE_OPENAI_API_KEY"),
            "azure_openai_chat_deployment": _require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            "azure_openai_embedding_deployment": _require(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            ),
            # Optional — falls back to a sensible default
            "azure_openai_api_version": os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-06-01"
            ),
            "apim_subscription_key": _require("APIM_SUBSCRIPTION_KEY"),
            "azure_search_endpoint": _require("AZURE_SEARCH_ENDPOINT"),
            "azure_search_key": _require("AZURE_SEARCH_KEY"),
            # Optional — defaults to "rag-index"
            "azure_search_index_name": os.getenv(
                "AZURE_SEARCH_INDEX_NAME", "rag-index"
            ),
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                # Bypass frozen=True to set values during initialisation
                object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        """Redact secrets so they don't leak into logs or tracebacks."""
        return (
            f"Settings("
            f"azure_openai_endpoint={self.azure_openai_endpoint!r}, "
            f"azure_openai_api_key='***', "
            f"azure_openai_chat_deployment={self.azure_openai_chat_deployment!r}, "
            f"azure_openai_embedding_deployment={self.azure_openai_embedding_deployment!r}, "
            f"azure_openai_api_version={self.azure_openai_api_version!r}, "
            f"apim_subscription_key='***', "
            f"azure_search_endpoint={self.azure_search_endpoint!r}, "
            f"azure_search_key='***', "
            f"azure_search_index_name={self.azure_search_index_name!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton (created on first call)."""
    return Settings()
