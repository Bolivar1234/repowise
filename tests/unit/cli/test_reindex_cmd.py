"""Tests for the reindex CLI command internals."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from repowise.cli.commands import reindex_cmd
from repowise.core.persistence.vector_store._base import EMBED_TEXT_MAX_CHARS, iter_embed_chunks


class _DummyEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class _EmptyResult:
    def scalars(self) -> _EmptyResult:
        return self

    def all(self) -> list[Any]:
        return []


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[Any]:
        return self._rows


class _Session:
    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, _stmt: object) -> _EmptyResult:
        return _EmptyResult()


def _sessionmaker(*_args: object, **_kwargs: object):
    return _Session


class _Page:
    id = "page-1"
    title = "Huge Page"
    content = "x" * (EMBED_TEXT_MAX_CHARS + 5000)
    page_type = "file_page"
    target_path = "huge.py"


class _ReindexSession(_Session):
    calls = 0

    async def execute(self, _stmt: object) -> _Result:
        type(self).calls += 1
        if type(self).calls == 1:
            return _Result([_Page()])
        return _Result([])


def _reindex_sessionmaker(*_args: object, **_kwargs: object):
    _ReindexSession.calls = 0
    return _ReindexSession


class _RecordingVectorStore:
    calls: list[list[tuple[str, str, dict]]] = []
    embedded_text_lengths: list[int] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).calls = []
        type(self).embedded_text_lengths = []

    async def embed_batch(self, items: list[tuple[str, str, dict]]) -> None:
        type(self).calls.append(items)
        for _chunk, texts in iter_embed_chunks(items):
            type(self).embedded_text_lengths.extend(len(text) for text in texts)

    async def close(self) -> None:
        return None


async def test_reindex_uses_shared_database_engine(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'wiki.db'}"
    created: dict[str, object] = {}

    def fake_create_engine(url: str):
        engine = _DummyEngine()
        created["url"] = url
        created["engine"] = engine
        return engine

    async def fake_init_db(engine: object) -> None:
        created["init_engine"] = engine

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(reindex_cmd, "get_db_url_for_repo", lambda _repo_path: db_url)
    monkeypatch.setattr(
        "repowise.core.persistence.database.create_engine",
        fake_create_engine,
    )
    monkeypatch.setattr("repowise.core.persistence.database.init_db", fake_init_db)
    monkeypatch.setattr("sqlalchemy.ext.asyncio.async_sessionmaker", _sessionmaker)

    await reindex_cmd._reindex(tmp_path, "openai", batch_size=20)

    assert created["url"] == db_url
    assert created["init_engine"] is created["engine"]
    assert created["engine"].disposed is True


async def test_reindex_uses_capped_batch_embedding_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'wiki.db'}"

    async def fake_init_db(_engine: object) -> None:
        return None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(reindex_cmd, "get_db_url_for_repo", lambda _repo_path: db_url)
    monkeypatch.setattr(
        "repowise.core.persistence.database.create_engine",
        lambda _url: _DummyEngine(),
    )
    monkeypatch.setattr("repowise.core.persistence.database.init_db", fake_init_db)
    monkeypatch.setattr("sqlalchemy.ext.asyncio.async_sessionmaker", _reindex_sessionmaker)
    monkeypatch.setattr(
        "repowise.core.persistence.vector_store.LanceDBVectorStore",
        _RecordingVectorStore,
    )

    await reindex_cmd._reindex(tmp_path, "openai", batch_size=20)

    assert len(_RecordingVectorStore.calls) == 1
    assert _RecordingVectorStore.calls[0][0][0] == "page-1"
    assert len(_RecordingVectorStore.calls[0][0][1]) > EMBED_TEXT_MAX_CHARS
    assert _RecordingVectorStore.embedded_text_lengths == [EMBED_TEXT_MAX_CHARS]


def test_reindex_auto_uses_saved_embedder_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("REPOWISE_EMBEDDER", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("REPOWISE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_DIMS", raising=False)
    monkeypatch.delenv("REPOWISE_EMBEDDING_DIMS", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(
        reindex_cmd,
        "load_config",
        lambda _repo_path: {
            "embedder": "ollama",
            "embedding_model": "nomic-embed-text",
            "embedding_dims": 768,
            "ollama_base_url": "http://127.0.0.1:11434",
        },
    )

    resolved = reindex_cmd._resolve_reindex_embedder(tmp_path, "auto")

    assert resolved == "ollama"
    assert os.environ["OLLAMA_EMBEDDING_MODEL"] == "nomic-embed-text"
    assert os.environ["REPOWISE_EMBEDDING_MODEL"] == "nomic-embed-text"
    assert os.environ["OLLAMA_EMBEDDING_DIMS"] == "768"
    assert os.environ["REPOWISE_EMBEDDING_DIMS"] == "768"
    assert os.environ["OLLAMA_BASE_URL"] == "http://127.0.0.1:11434"


def test_reindex_env_overrides_saved_embedding_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("REPOWISE_EMBEDDING_MODEL", "env-model")
    monkeypatch.setattr(
        reindex_cmd,
        "load_config",
        lambda _repo_path: {
            "embedder": "ollama",
            "embedding_model": "config-model",
        },
    )

    resolved = reindex_cmd._resolve_reindex_embedder(tmp_path, "ollama")

    assert resolved == "ollama"
    assert os.environ["OLLAMA_EMBEDDING_MODEL"] == "env-model"
    assert os.environ["REPOWISE_EMBEDDING_MODEL"] == "env-model"
