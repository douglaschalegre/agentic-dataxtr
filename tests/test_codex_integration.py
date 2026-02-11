"""Tests for OpenAI Codex provider integration."""

import json
import sys
from types import SimpleNamespace

import pytest

from dataxtr.graph.nodes import _resolve_preferred_provider
from dataxtr.graph.state import create_initial_state
from dataxtr.models.codex_auth import CodexAuth, resolve_codex_auth
from dataxtr.models.config import MODEL_REGISTRY, ModelProvider
from dataxtr.models.router import ModelRouter


def test_resolve_codex_auth_prefers_env(monkeypatch):
    """Environment token should override file-based auth."""
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "env-token")
    monkeypatch.setenv("OPENAI_CODEX_ACCOUNT_ID", "env-account")

    auth = resolve_codex_auth()

    assert auth.access_token == "env-token"
    assert auth.account_id == "env-account"


def test_resolve_codex_auth_reads_auth_file(tmp_path, monkeypatch):
    """Fallback to Codex auth.json when env token is absent."""
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_CODEX_ACCOUNT_ID", raising=False)

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "file-token",
                    "account_id": "file-account",
                }
            }
        )
    )
    monkeypatch.setenv("OPENAI_CODEX_AUTH_FILE", str(auth_file))

    auth = resolve_codex_auth()

    assert auth.access_token == "file-token"
    assert auth.account_id == "file-account"


def test_resolve_codex_auth_raises_without_token(tmp_path, monkeypatch):
    """Invalid auth file should produce a clear error."""
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {}}))
    monkeypatch.setenv("OPENAI_CODEX_AUTH_FILE", str(auth_file))

    with pytest.raises(ValueError, match="access_token"):
        resolve_codex_auth()


def test_resolve_preferred_provider_uses_state_then_env(monkeypatch):
    """State override should win over DEFAULT_LLM_PROVIDER env var."""
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "ollama")

    provider = _resolve_preferred_provider({"preferred_provider": "openai_codex"})  # type: ignore[arg-type]

    assert provider == ModelProvider.OPENAI_CODEX


def test_create_initial_state_accepts_preferred_provider(sample_invoice_schema):
    """Initial state should carry preferred provider override."""
    state = create_initial_state(
        document_path="/tmp/example.pdf",
        document_type="pdf",
        schema_fields=sample_invoice_schema,
        preferred_provider="openai_codex",
    )

    assert state["preferred_provider"] == "openai_codex"


def test_model_registry_includes_openai_codex_provider():
    """Codex models must be available for router selection."""
    providers = {config.provider for config in MODEL_REGISTRY.values()}
    assert ModelProvider.OPENAI_CODEX in providers


def test_router_instantiates_chatopenai_for_codex(monkeypatch):
    """Router should build ChatOpenAI with Codex endpoint and headers."""

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.setattr(
        "dataxtr.models.codex_auth.resolve_codex_auth",
        lambda: CodexAuth(access_token="token-123", account_id="acct-123"),
    )
    monkeypatch.setenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")

    router = ModelRouter()
    model = router.get_chat_model(MODEL_REGISTRY["openai-codex-gpt-5.3-codex"])

    assert isinstance(model, FakeChatOpenAI)
    assert model.kwargs["api_key"] == "token-123"
    assert model.kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert model.kwargs["default_headers"] == {"ChatGPT-Account-Id": "acct-123"}
