"""Tests for config.py — settings loading, validation, and defaults."""

import os
from unittest.mock import patch, MagicMock

import pytest


def _make_env(**overrides):
    """Build a complete set of valid env vars with optional overrides."""
    base = {
        "AZURE_TENANT_ID": "test-tenant-id",
        "AZURE_CLIENT_ID": "test-client-id",
        "AZURE_CLIENT_SECRET": "test-client-secret",
        "APIM_SCOPE": "api://test-app-id/.default",
        "APIM_ENDPOINT": "https://test.azure-api.net",
        "AZURE_KEYVAULT_NAME": "",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
        "AZURE_SEARCH_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
    }
    base.update(overrides)
    return base


@patch("config._build_credential")
def test_settings_loads_all_vars(mock_cred):
    mock_cred.return_value = MagicMock()
    env = _make_env()
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

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
    assert s.credential is mock_cred.return_value


@patch("config._build_credential")
def test_settings_missing_required_var(mock_cred):
    mock_cred.return_value = MagicMock()
    env = _make_env()
    del env["AZURE_CLIENT_SECRET"]
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        with pytest.raises(EnvironmentError, match="AZURE_CLIENT_SECRET"):
            Settings()


@patch("config._build_credential")
def test_settings_defaults(mock_cred):
    mock_cred.return_value = MagicMock()
    env = _make_env()
    del env["AZURE_OPENAI_API_VERSION"]
    del env["AZURE_SEARCH_INDEX_NAME"]
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

    assert s.azure_openai_api_version == "2024-06-01"
    assert s.azure_search_index_name == "rag-index"


@patch("config._build_credential")
def test_settings_repr_redacts_secrets(mock_cred):
    mock_cred.return_value = MagicMock()
    env = _make_env()
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()
        text = repr(s)

    assert "test-client-secret" not in text
    assert "test-search-key" not in text
    assert "***" in text
