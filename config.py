"""Centralised application settings loaded from environment variables.

Reads Azure AD (Entra ID), APIM, Key Vault, and Azure AI Search configuration
from a .env file (via python-dotenv) and exposes a cached Settings singleton
via get_settings().

Authentication flow:
1. Bootstrap vars loaded from .env (Tenant ID, Client ID, Client Secret, etc.)
2. ClientSecretCredential created for Azure AD OAuth2
3. Optional: fetch additional secrets from Azure Key Vault (falls back to env)
4. credential object stored on Settings for Bearer token acquisition
"""

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _require(var: str) -> str:
    """Read an environment variable or raise if it is missing/empty."""
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value


def _build_credential(tenant_id: str, client_id: str, client_secret: str):
    """Create an Azure AD ClientSecretCredential for OAuth2 client credentials flow."""
    from azure.identity import ClientSecretCredential

    return ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )


def _fetch_keyvault_secret(credential, vault_name: str, secret_name: str) -> Optional[str]:
    """Try to fetch a single secret from Key Vault. Returns None on failure."""
    try:
        from azure.keyvault.secrets import SecretClient

        vault_url = f"https://{vault_name}.vault.azure.net"
        client = SecretClient(vault_url=vault_url, credential=credential)
        return client.get_secret(secret_name).value
    except Exception as exc:
        logger.debug("Could not fetch '%s' from Key Vault: %s", secret_name, exc)
        return None


@dataclass(frozen=True)
class Settings:
    """Immutable container for all Azure service configuration.

    Fields default to None so the class can be imported without triggering
    env-var validation. Actual values are resolved in __post_init__.
    """

    # OAuth2 / Entra ID
    azure_tenant_id: str = field(default=None)
    azure_client_id: str = field(default=None)
    azure_client_secret: str = field(default=None)
    apim_scope: str = field(default=None)

    # APIM
    apim_endpoint: str = field(default=None)

    # Key Vault
    azure_keyvault_name: str = field(default=None)

    # Azure OpenAI (non-secret deployment config)
    azure_openai_chat_deployment: str = field(default=None)
    azure_openai_embedding_deployment: str = field(default=None)
    azure_openai_api_version: str = field(default=None)

    # Azure AI Search
    azure_search_endpoint: str = field(default=None)
    azure_search_key: str = field(default=None)
    azure_search_index_name: str = field(default=None)

    # Runtime objects (not from env)
    credential: object = field(default=None)

    def __post_init__(self):
        defaults = {
            "azure_tenant_id": _require("AZURE_TENANT_ID"),
            "azure_client_id": _require("AZURE_CLIENT_ID"),
            "azure_client_secret": _require("AZURE_CLIENT_SECRET"),
            "apim_scope": _require("APIM_SCOPE"),
            "apim_endpoint": _require("APIM_ENDPOINT"),
            "azure_keyvault_name": os.getenv("AZURE_KEYVAULT_NAME", ""),
            "azure_openai_chat_deployment": _require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            "azure_openai_embedding_deployment": _require(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            ),
            "azure_openai_api_version": os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-06-01"
            ),
            "azure_search_endpoint": _require("AZURE_SEARCH_ENDPOINT"),
            "azure_search_key": _require("AZURE_SEARCH_KEY"),
            "azure_search_index_name": os.getenv(
                "AZURE_SEARCH_INDEX_NAME", "rag-index"
            ),
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)

        # Build OAuth2 credential
        credential = _build_credential(
            self.azure_tenant_id,
            self.azure_client_id,
            self.azure_client_secret,
        )
        object.__setattr__(self, "credential", credential)

        # Optionally fetch secrets from Key Vault (falls back to env vars)
        if self.azure_keyvault_name:
            search_key = _fetch_keyvault_secret(
                credential, self.azure_keyvault_name, "azure-search-key"
            )
            if search_key:
                object.__setattr__(self, "azure_search_key", search_key)

    def __repr__(self) -> str:
        """Redact secrets so they don't leak into logs or tracebacks."""
        return (
            f"Settings("
            f"azure_tenant_id={self.azure_tenant_id!r}, "
            f"azure_client_id={self.azure_client_id!r}, "
            f"azure_client_secret='***', "
            f"apim_scope={self.apim_scope!r}, "
            f"apim_endpoint={self.apim_endpoint!r}, "
            f"azure_keyvault_name={self.azure_keyvault_name!r}, "
            f"azure_openai_chat_deployment={self.azure_openai_chat_deployment!r}, "
            f"azure_openai_embedding_deployment={self.azure_openai_embedding_deployment!r}, "
            f"azure_openai_api_version={self.azure_openai_api_version!r}, "
            f"azure_search_endpoint={self.azure_search_endpoint!r}, "
            f"azure_search_key='***', "
            f"azure_search_index_name={self.azure_search_index_name!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton (created on first call)."""
    return Settings()
