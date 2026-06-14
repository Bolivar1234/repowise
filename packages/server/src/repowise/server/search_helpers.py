"""Helpers for search fan-out across workspace repos.

These live in a separate module so they can be reused by other routers
(chat, MCP-over-HTTP, future workspace endpoints) without each one
re-implementing LanceDB rehydration / lazy caching.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("repowise.server.search_helpers")

# Asyncio lock per repo_id, used to prevent two concurrent semantic
# search requests from both opening the LanceDB store. The first one in
# allocates; everyone else awaits the cached result.
_vector_store_locks: dict[str, asyncio.Lock] = {}


async def resolve_workspace_vector_store(app, repo_id: str) -> Any | None:
    """Return a vector store for the given workspace ``repo_id``.

    Cached on ``app.state.workspace_vector_stores`` after first
    resolution. Returns ``None`` if:
      - We're not in workspace mode.
      - The repo_id isn't in the workspace config.
      - The repo has no ``lancedb/`` directory on disk yet.

    The store uses the same embedder as the primary vector store, so a
    workspace built with the gemini embedder keeps its embeddings
    compatible across fan-out queries.
    """
    cache: dict | None = getattr(app.state, "workspace_vector_stores", None)
    if cache is None:
        return None
    if repo_id in cache:
        return cache[repo_id]

    lock = _vector_store_locks.setdefault(repo_id, asyncio.Lock())
    async with lock:
        # Double-checked locking — another coroutine may have populated
        # the cache while we waited on the lock.
        if repo_id in cache:
            return cache[repo_id]

        ws_config = getattr(app.state, "workspace_config", None)
        ws_root = getattr(app.state, "workspace_root", None)
        if ws_config is None or ws_root is None:
            return None

        # Locate the repo's directory by repo_id. We don't carry an
        # alias→repo_id map directly, so scan once. Workspace sizes are
        # small (typically <20 repos), so the linear scan is fine.
        import sqlite3

        repo_path: Path | None = None
        ws_root_path = Path(ws_root)
        for entry in ws_config.repos:
            candidate = (ws_root_path / entry.path).resolve()
            db_path = candidate / ".repowise" / "wiki.db"
            if not db_path.exists():
                continue
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    row = conn.execute(
                        "SELECT id FROM repositories LIMIT 1"
                    ).fetchone()
                if row and row[0] == repo_id:
                    repo_path = candidate
                    break
            except Exception:
                continue
        if repo_path is None:
            return None

        lance_dir = repo_path / ".repowise" / "lancedb"
        if not lance_dir.is_dir():
            return None

        # Build the store with the same embedder used by the primary
        # vector store. Falls back to mock when the primary is also
        # mock — keeps unit tests deterministic.
        try:
            from repowise.core.persistence.vector_store import LanceDBVectorStore

            embedder = _resolve_embedder(app)
            store = LanceDBVectorStore(str(lance_dir), embedder=embedder)
            cache[repo_id] = store
            return store
        except Exception:
            logger.debug(
                "lancedb_open_failed",
                extra={"repo_id": repo_id, "path": str(lance_dir)},
                exc_info=True,
            )
            return None


def _resolve_embedder(app):
    """Pull the same embedder the primary vector store was built with.

    ``InMemoryVectorStore`` and ``LanceDBVectorStore`` both expose
    ``_embedder`` (set in __init__). Falls back to a fresh mock when the
    primary store doesn't expose one — keeps semantic search "working"
    against LanceDB stores built with the mock embedder during tests.
    """
    from repowise.core.providers.embedding.base import MockEmbedder

    primary_vs = getattr(app.state, "vector_store", None)
    embedder = getattr(primary_vs, "_embedder", None) if primary_vs is not None else None
    if embedder is not None and not isinstance(embedder, MockEmbedder):
        return embedder

    configured = os.environ.get("REPOWISE_EMBEDDER", "").strip().lower()
    if configured and configured != "mock":
        try:
            from repowise.server.app import _build_embedder

            resolved = _build_embedder()
            if not isinstance(resolved, MockEmbedder):
                return resolved
        except Exception:
            logger.debug("configured_embedder_resolution_failed", exc_info=True)

    return embedder or MockEmbedder()


async def close_workspace_vector_stores(app) -> None:
    """Close every cached workspace vector store. Called on shutdown."""
    cache: dict | None = getattr(app.state, "workspace_vector_stores", None)
    if not cache:
        return
    for store in list(cache.values()):
        with contextlib.suppress(Exception):
            await store.close()
    cache.clear()
