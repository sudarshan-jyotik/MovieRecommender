"""Lazy semantic search over precomputed movie embeddings (stub until Deliverable 3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "data" / "movie_embeddings.npz"


def is_semantic_enabled() -> bool:
    """Return True when semantic search is enabled via ENABLE_SEMANTIC=1."""
    return os.getenv("ENABLE_SEMANTIC", "1").strip().lower() in {"1", "true", "yes", "on"}


def semantic_search_movies(
    description: str,
    limit: int = 8,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
) -> list[dict[str, Any]]:
    """Return top-K semantically matching movies or raise SemanticUnavailableError."""
    raise RuntimeError("Semantic search is not initialized yet.")
