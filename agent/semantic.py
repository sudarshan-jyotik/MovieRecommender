"""Lazy semantic search over precomputed float16 movie embeddings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "data" / "movie_embeddings.npz"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class SemanticUnavailableError(RuntimeError):
    """Raised when semantic search is disabled, unbuilt, or model loading fails."""


def is_semantic_enabled() -> bool:
    """Return True when semantic search is enabled via ENABLE_SEMANTIC=1."""
    return os.getenv("ENABLE_SEMANTIC", "1").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=2)
def _load_embeddings_index(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    if not path.is_file():
        raise SemanticUnavailableError(
            f"Missing embeddings artifact at {path}. Run `python scripts/build_embeddings.py`."
        )
    with np.load(path) as data:
        movie_ids = np.asarray(data["movie_ids"], dtype=np.int64)
        matrix = np.asarray(data["embeddings"], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / norms
    return movie_ids, normalized


@lru_cache(maxsize=1)
def _get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise SemanticUnavailableError("fastembed is not installed") from exc

    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise SemanticUnavailableError(f"Failed to load embedding model {model_name}: {exc}") from exc


def embed_query(description: str, embedder: Any | None = None) -> np.ndarray:
    """Embed a natural-language query into a unit-norm float32 vector."""
    model = embedder if embedder is not None else _get_embedder()
    vectors = list(model.embed([description]))
    if not vectors:
        raise SemanticUnavailableError("Embedding model returned no vector for query")
    vec = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec


def semantic_search_movies(
    description: str,
    limit: int = 8,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    embedder: Any | None = None,
) -> list[dict[str, Any]]:
    """Return top-K semantically matching movie_id/similarity dicts."""
    if not is_semantic_enabled():
        raise SemanticUnavailableError("Semantic search is disabled via ENABLE_SEMANTIC")
    if not description or not description.strip():
        raise ValueError("description must not be empty")

    movie_ids, normalized_matrix = _load_embeddings_index(str(embeddings_path.resolve()))
    query_vec = embed_query(description.strip(), embedder=embedder)

    scores = normalized_matrix @ query_vec
    k = max(1, min(int(limit), len(movie_ids)))
    part_idx = np.argpartition(scores, -k)[-k:]
    top_idx = part_idx[np.argsort(-scores[part_idx])]

    return [
        {
            "movie_id": int(movie_ids[idx]),
            "similarity": float(scores[idx]),
        }
        for idx in top_idx.tolist()
    ]
