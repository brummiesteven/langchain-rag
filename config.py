"""Centralised application settings loaded from environment variables.

This is the single source of truth for all configuration in the app.
Every Azure service credential and deployment name flows through here.

HOW IT WORKS:
- On import, python-dotenv reads your .env file and populates os.environ.
- When get_settings() is called for the first time, it creates a Settings
  dataclass that pulls values from os.environ.
- The Settings object also builds an OAuth2 credential (ClientSecretCredential)
  which is used later by rag/llm.py to get Bearer tokens for APIM.
- If a Key Vault name is configured, it will attempt to fetch secrets from
  Azure Key Vault (e.g. the Search admin key), falling back to env vars if
  Key Vault is unreachable (useful for local dev without Key Vault access).
- The Settings singleton is cached via @lru_cache so it's only created once.

AUTHENTICATION FLOW (how the app authenticates to Azure services):
1. .env provides bootstrap credentials: Tenant ID, Client ID, Client Secret.
   These identify the app registration in Azure AD (Entra ID).
2. A ClientSecretCredential is created from those three values. This object
   can request OAuth2 access tokens for any Azure resource the app reg has
   permissions for.
3. (Optional) If AZURE_KEYVAULT_NAME is set, the credential is used to
   connect to Key Vault and fetch secrets stored there (e.g. the Search key).
   This avoids hardcoding secrets in .env for production deployments.
4. The credential object is stored on the Settings dataclass so that
   rag/llm.py can call credential.get_token() to get Bearer tokens for APIM.
"""

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Read .env file into os.environ. This must happen at module level so that
# all subsequent os.getenv() calls can see the values. If .env doesn't exist
# (e.g. in Docker where env vars are injected directly), this is a no-op.
load_dotenv()

# Logger for this module — used to log Key Vault fetch failures at DEBUG level
# so they don't spam the console in local dev (where Key Vault is often absent).
logger = logging.getLogger(__name__)


def _require(var: str) -> str:
    """Read an environment variable or raise if it is missing/empty.

    This is used for variables that the app absolutely cannot function without
    (e.g. AZURE_TENANT_ID). If any required var is missing, the app fails fast
    with a clear error message rather than crashing later with a confusing one.
    """
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value


def _build_credential(tenant_id: str, client_id: str, client_secret: str):
    """Create an Azure AD ClientSecretCredential for OAuth2 client credentials flow.

    This is the core authentication object for the entire app. It uses the
    "client credentials" OAuth2 grant type, which is designed for service-to-service
    auth (no user interaction needed).

    The import is inside the function (lazy import) so that the azure-identity
    package isn't loaded until actually needed. This allows tests to import
    config.py and mock this function without needing azure-identity installed.

    Args:
        tenant_id: The Azure AD tenant (directory) ID — identifies which Azure AD
                   instance to authenticate against.
        client_id: The application (client) ID — identifies this specific app
                   registration in Azure AD.
        client_secret: A secret string generated in the app registration's
                      "Certificates & secrets" blade. Acts as the app's password.

    Returns:
        A ClientSecretCredential that can generate OAuth2 access tokens via
        credential.get_token(scope).
    """
    from azure.identity import ClientSecretCredential

    return ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )


def _fetch_keyvault_secret(credential, vault_name: str, secret_name: str) -> Optional[str]:
    """Try to fetch a single secret from Azure Key Vault.

    Key Vault is Azure's managed secret store. Instead of putting sensitive values
    (like API keys) directly in env vars or config files, you store them in Key Vault
    and fetch them at runtime. This is more secure because:
    - Secrets are encrypted at rest and in transit
    - Access is controlled via Azure RBAC (role-based access control)
    - You get audit logs of who accessed what secret and when

    This function is intentionally lenient — if Key Vault is unreachable (e.g.
    running locally without VPN, or Key Vault doesn't exist yet), it returns None
    and the caller falls back to the env var value. This makes local dev easy.

    Args:
        credential: The ClientSecretCredential created in _build_credential().
                   Used to authenticate to Key Vault.
        vault_name: Just the vault name (e.g. "my-vault"), not the full URL.
                   We construct the URL as https://{vault_name}.vault.azure.net.
        secret_name: The name of the secret to fetch (e.g. "azure-search-key").

    Returns:
        The secret value as a string, or None if the fetch failed for any reason.
    """
    try:
        from azure.keyvault.secrets import SecretClient

        # Every Key Vault has a URL in this format
        vault_url = f"https://{vault_name}.vault.azure.net"
        client = SecretClient(vault_url=vault_url, credential=credential)
        return client.get_secret(secret_name).value
    except Exception as exc:
        # Log at DEBUG so it doesn't clutter the console. In local dev, Key Vault
        # is usually not available and that's expected — we fall back to env vars.
        logger.debug("Could not fetch '%s' from Key Vault: %s", secret_name, exc)
        return None


@dataclass(frozen=True)
class Settings:
    """Immutable container for all Azure service configuration.

    WHY A FROZEN DATACLASS?
    - frozen=True makes the object immutable after creation. This prevents
      accidental mutation of settings at runtime, which would be a bug.
    - Dataclass gives us __init__, __eq__, and field declarations for free.

    WHY DO ALL FIELDS DEFAULT TO None?
    - So you can write `from config import Settings` without triggering env var
      validation. Validation only happens when you actually create an instance
      (Settings()), which calls __post_init__.
    - This is crucial for tests — they can import the class, mock the env, then
      create an instance. If fields had no defaults, the import itself would fail.

    HOW __post_init__ WORKS WITH frozen=True:
    - Normally you can't set attributes on a frozen dataclass (it raises
      FrozenInstanceError). But __post_init__ runs during __init__, and we use
      object.__setattr__() to bypass the frozen check. This is a standard Python
      pattern for "set values once during initialisation, then freeze".
    """

    # ─── OAuth2 / Entra ID ────────────────────────────────────────────────
    # These three values come from the app registration in Azure AD (Entra ID).
    # Together they let the app authenticate as a service principal.
    azure_tenant_id: str = field(default=None)       # Which Azure AD tenant
    azure_client_id: str = field(default=None)        # Which app registration
    azure_client_secret: str = field(default=None)    # The app's password/secret

    # The OAuth2 scope to request when getting a token for APIM.
    # Format is typically: api://<app-id>/.default
    # The ".default" suffix means "give me all permissions this app is allowed".
    apim_scope: str = field(default=None)

    # ─── APIM ─────────────────────────────────────────────────────────────
    # The base URL of the Azure API Management gateway. All LLM and embedding
    # requests go through this gateway (not directly to Azure OpenAI).
    # Example: https://my-company-apim.azure-api.net
    apim_endpoint: str = field(default=None)

    # ─── Key Vault ────────────────────────────────────────────────────────
    # Optional. If set, the app will try to fetch secrets from this Key Vault
    # at startup. If empty/unset, all secrets come from env vars instead.
    azure_keyvault_name: str = field(default=None)

    # ─── Azure OpenAI (non-secret deployment config) ──────────────────────
    # These are just deployment names — not secrets. They tell the Azure OpenAI
    # service which model to use. Set in the Azure OpenAI Studio when you deploy.
    azure_openai_chat_deployment: str = field(default=None)      # e.g. "gpt-4o"
    azure_openai_embedding_deployment: str = field(default=None)  # e.g. "text-embedding-ada-002"
    azure_openai_api_version: str = field(default=None)           # API version string

    # ─── Azure AI Search ──────────────────────────────────────────────────
    # Used for the vector store (where document chunks + embeddings are stored).
    # Search is accessed DIRECTLY (not through APIM) using its own endpoint + key.
    azure_search_endpoint: str = field(default=None)   # e.g. https://my-search.search.windows.net
    azure_search_key: str = field(default=None)        # Admin key for full read/write access
    azure_search_index_name: str = field(default=None)  # Name of the search index

    # ─── Runtime objects (not from env vars) ──────────────────────────────
    # The ClientSecretCredential object, built from tenant/client/secret above.
    # Stored here so that rag/llm.py can call credential.get_token() without
    # needing to rebuild it. Typed as `object` to avoid importing azure-identity
    # at the module level (keeps imports lazy for tests).
    credential: object = field(default=None)

    def __post_init__(self):
        """Resolve all field values from environment variables.

        This runs automatically when Settings() is called. It reads env vars,
        validates required ones, sets defaults for optional ones, builds the
        OAuth2 credential, and optionally fetches secrets from Key Vault.
        """
        # Map each field name to its value from the environment.
        # _require() raises EnvironmentError if the var is missing.
        # os.getenv() with a default is used for optional vars.
        defaults = {
            "azure_tenant_id": _require("AZURE_TENANT_ID"),
            "azure_client_id": _require("AZURE_CLIENT_ID"),
            "azure_client_secret": _require("AZURE_CLIENT_SECRET"),
            "apim_scope": _require("APIM_SCOPE"),
            "apim_endpoint": _require("APIM_ENDPOINT"),
            # Key Vault name is optional — empty string means "don't use Key Vault"
            "azure_keyvault_name": os.getenv("AZURE_KEYVAULT_NAME", ""),
            "azure_openai_chat_deployment": _require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            "azure_openai_embedding_deployment": _require(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            ),
            # API version defaults to a known-good value if not explicitly set
            "azure_openai_api_version": os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-06-01"
            ),
            "azure_search_endpoint": _require("AZURE_SEARCH_ENDPOINT"),
            "azure_search_key": _require("AZURE_SEARCH_KEY"),
            "azure_search_index_name": os.getenv(
                "AZURE_SEARCH_INDEX_NAME", "rag-index"
            ),
        }

        # Set each field value. We only set if the field is still None (its default).
        # This allows tests to pass explicit values via Settings(azure_tenant_id="x")
        # which would skip the env-var lookup for that field.
        # object.__setattr__ is needed because the dataclass is frozen.
        for key, value in defaults.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)

        # Build the OAuth2 credential object from the three identity values.
        # This credential is what we'll use to get Bearer tokens for APIM,
        # and also to authenticate to Key Vault if configured.
        credential = _build_credential(
            self.azure_tenant_id,
            self.azure_client_id,
            self.azure_client_secret,
        )
        object.__setattr__(self, "credential", credential)

        # If a Key Vault name is configured, try to fetch secrets from it.
        # Currently we only fetch the Azure Search key from Key Vault.
        # If the fetch fails (e.g. no network, wrong permissions), we keep
        # whatever value was already loaded from the env var.
        if self.azure_keyvault_name:
            search_key = _fetch_keyvault_secret(
                credential, self.azure_keyvault_name, "azure-search-key"
            )
            if search_key:
                object.__setattr__(self, "azure_search_key", search_key)

    def __repr__(self) -> str:
        """Custom string representation that redacts secrets.

        If someone accidentally logs or prints the Settings object (e.g. in a
        traceback), we don't want secrets leaking into log files or consoles.
        The client_secret and search_key are replaced with '***'.
        """
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
    """Return a cached Settings singleton (created on first call).

    @lru_cache(maxsize=1) means the first call creates a Settings instance and
    caches it. Every subsequent call returns the same instance without re-reading
    env vars or re-creating the credential. This is a simple singleton pattern.

    The cache means env var changes after the first call are NOT picked up.
    Restart the app to pick up new env var values.
    """
    return Settings()
