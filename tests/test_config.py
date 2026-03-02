"""Tests for config.py — settings loading, validation, and defaults.

HOW THESE TESTS WORK:
  Each test creates a fake set of environment variables using _make_env(),
  then uses patch.dict(os.environ, ..., clear=True) to REPLACE the real
  environment with only our test values. clear=True means the real env vars
  (like PATH, HOME, etc.) are temporarily removed — this ensures tests don't
  accidentally read real Azure credentials.

  We also mock _build_credential so that tests don't need the azure-identity
  package to actually create a real ClientSecretCredential (which would try to
  contact Azure AD). The mock returns a MagicMock object that we can check
  was assigned to settings.credential.

WHY `from config import Settings` INSIDE EACH TEST?
  Python caches modules after the first import. Since config.py calls
  load_dotenv() at module level, we import Settings inside each test (after
  setting up the mock environment) to ensure the env vars are correct.
  In practice, since the module is already cached, this just grabs the
  already-imported Settings class — but doing it inside the patch context
  is a safety habit.
"""

import os
from unittest.mock import patch, MagicMock

import pytest


def _make_env(**overrides):
    """Build a complete set of valid env vars with optional overrides.

    Returns a dict that has every env var the Settings class needs.
    Pass keyword arguments to override specific values or add extras.

    Example:
        _make_env()  → all defaults (valid config)
        _make_env(AZURE_TENANT_ID="custom")  → overrides just the tenant ID
    """
    base = {
        # OAuth2 credentials — identify the app in Azure AD
        "AZURE_TENANT_ID": "test-tenant-id",
        "AZURE_CLIENT_ID": "test-client-id",
        "AZURE_CLIENT_SECRET": "test-client-secret",
        # OAuth2 scope for APIM — what resource we're requesting access to
        "APIM_SCOPE": "api://test-app-id/.default",
        # APIM gateway URL
        "APIM_ENDPOINT": "https://test.azure-api.net",
        # Empty string = Key Vault disabled (won't try to fetch secrets)
        "AZURE_KEYVAULT_NAME": "",
        # Azure OpenAI deployment names
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        # Azure AI Search config
        "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
        "AZURE_SEARCH_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
    }
    base.update(overrides)
    return base


# ─── Test: All variables load correctly ───────────────────────────────────────
@patch("config._build_credential")
def test_settings_loads_all_vars(mock_cred):
    """Verify that every env var is correctly read into the Settings fields."""
    # _build_credential is mocked so we don't need real Azure AD credentials.
    # The mock returns a MagicMock that we can later verify was stored on Settings.
    mock_cred.return_value = MagicMock()
    env = _make_env()

    # patch.dict replaces os.environ with ONLY our test values (clear=True)
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

    # Verify every field matches the test env var value
    assert s.azure_tenant_id == "test-tenant-id"
    assert s.azure_client_id == "test-client-id"
    assert s.azure_client_secret == "test-client-secret"
    assert s.apim_scope == "api://test-app-id/.default"
    assert s.apim_endpoint == "https://test.azure-api.net"
    assert s.azure_openai_chat_deployment == "gpt-4o"
    assert s.azure_openai_embedding_deployment == "text-embedding-ada-002"
    assert s.azure_search_endpoint == "https://test.search.windows.net"
    assert s.azure_search_key == "test-search-key"
    assert s.azure_search_index_name == "test-index"
    # The credential should be the mock object returned by _build_credential
    assert s.credential is mock_cred.return_value


# ─── Test: Missing required var raises EnvironmentError ──────────────────────
@patch("config._build_credential")
def test_settings_missing_required_var(mock_cred):
    """Verify that a missing required env var raises EnvironmentError with the var name."""
    mock_cred.return_value = MagicMock()
    env = _make_env()
    # Remove a required variable to trigger the error
    del env["AZURE_CLIENT_SECRET"]

    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        # Should raise with a message containing the missing var name
        with pytest.raises(EnvironmentError, match="AZURE_CLIENT_SECRET"):
            Settings()


# ─── Test: Optional vars use sensible defaults ───────────────────────────────
@patch("config._build_credential")
def test_settings_defaults(mock_cred):
    """Verify that optional env vars fall back to their default values."""
    mock_cred.return_value = MagicMock()
    env = _make_env()
    # Remove the optional variables
    del env["AZURE_OPENAI_API_VERSION"]
    del env["AZURE_SEARCH_INDEX_NAME"]

    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

    # These should use their default values, not raise errors
    assert s.azure_openai_api_version == "2024-06-01"
    assert s.azure_search_index_name == "rag-index"


# ─── Test: repr() redacts secrets ────────────────────────────────────────────
@patch("config._build_credential")
def test_settings_repr_redacts_secrets(mock_cred):
    """Verify that repr() hides sensitive values so they don't leak into logs."""
    mock_cred.return_value = MagicMock()
    env = _make_env()

    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()
        text = repr(s)

    # The actual secret values should NOT appear in the repr output
    assert "test-client-secret" not in text  # azure_client_secret
    assert "test-search-key" not in text     # azure_search_key
    # They should be replaced with '***'
    assert "***" in text
