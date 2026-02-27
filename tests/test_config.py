"""Tests for config.py — settings loading, validation, and defaults."""

import os
from unittest.mock import patch

import pytest


def _make_env(**overrides):
    """Build a complete set of valid env vars with optional overrides."""
    base = {
        "AZURE_OPENAI_ENDPOINT": "https://test.azure-api.net",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        "APIM_SUBSCRIPTION_KEY": "test-sub-key",
        "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
        "AZURE_SEARCH_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
    }
    base.update(overrides)
    return base


def test_settings_loads_all_vars():
    env = _make_env()
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

    assert s.azure_openai_endpoint == "https://test.azure-api.net"
    assert s.azure_openai_api_key == "test-key"
    assert s.azure_openai_chat_deployment == "gpt-4o"
    assert s.azure_openai_embedding_deployment == "text-embedding-ada-002"
    assert s.apim_subscription_key == "test-sub-key"
    assert s.azure_search_endpoint == "https://test.search.windows.net"
    assert s.azure_search_key == "test-search-key"
    assert s.azure_search_index_name == "test-index"


def test_settings_missing_required_var():
    env = _make_env()
    del env["AZURE_OPENAI_API_KEY"]
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        with pytest.raises(EnvironmentError, match="AZURE_OPENAI_API_KEY"):
            Settings()


def test_settings_defaults():
    env = _make_env()
    del env["AZURE_OPENAI_API_VERSION"]
    del env["AZURE_SEARCH_INDEX_NAME"]
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()

    assert s.azure_openai_api_version == "2024-06-01"
    assert s.azure_search_index_name == "rag-index"


def test_settings_repr_redacts_secrets():
    env = _make_env()
    with patch.dict(os.environ, env, clear=True):
        from config import Settings

        s = Settings()
        text = repr(s)

    assert "test-key" not in text
    assert "test-sub-key" not in text
    assert "test-search-key" not in text
    assert "***" in text
