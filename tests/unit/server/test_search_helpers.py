from __future__ import annotations

from types import SimpleNamespace

from repowise.core.persistence.vector_store import InMemoryVectorStore
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.providers.embedding.ollama import OllamaEmbedder
from repowise.server.search_helpers import _resolve_embedder


def _app_with_embedder(embedder):
    return SimpleNamespace(
        state=SimpleNamespace(
            vector_store=InMemoryVectorStore(embedder=embedder),
        )
    )


def test_resolve_embedder_keeps_real_primary(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDER", "mock")
    primary = OllamaEmbedder(model="nomic-embed-text", base_url="http://127.0.0.1:11434")

    assert _resolve_embedder(_app_with_embedder(primary)) is primary


def test_resolve_embedder_keeps_mock_when_mock_configured(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDER", "mock")
    primary = MockEmbedder()

    assert _resolve_embedder(_app_with_embedder(primary)) is primary


def test_resolve_embedder_uses_configured_real_embedder_over_mock(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDER", "ollama")
    monkeypatch.setenv("REPOWISE_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "768")
    primary = MockEmbedder()

    resolved = _resolve_embedder(_app_with_embedder(primary))

    assert isinstance(resolved, OllamaEmbedder)
    assert resolved.dimensions == 768
