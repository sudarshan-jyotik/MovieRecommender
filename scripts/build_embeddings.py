"""Precompute float16 BAAI/bge-small-en-v1.5 embeddings from movie overview + tagline + genres."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOVIES_PATH = ROOT / "data" / "movies_clean.json.gz"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "movie_embeddings.npz"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def format_movie_document(movie: dict[str, Any]) -> str:
    """Build the text document used for semantic embedding of a movie."""
    title = str(movie.get("title") or "").strip()
    genres = ", ".join(str(g) for g in (movie.get("genres") or []) if g)
    tagline = str(movie.get("tagline") or "").strip()
    overview = str(movie.get("overview") or "").strip()

    parts = [f"Title: {title}"]
    if genres:
        parts.append(f"Genres: {genres}")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    if overview:
        parts.append(f"Overview: {overview}")
    return "\n".join(parts)


def build_embeddings(
    movies_path: Path = DEFAULT_MOVIES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> Path:
    """Generate float16 embeddings for all movies and save to compressed NPZ."""
    with gzip.open(movies_path, "rt", encoding="utf-8") as gz:
        movies: list[dict[str, Any]] = json.load(gz)

    movie_ids = np.asarray([int(m["movie_id"]) for m in movies], dtype=np.int32)
    docs = [format_movie_document(m) for m in movies]

    # Sort by length to minimize ONNX padding per batch, then restore original order
    order = sorted(range(len(docs)), key=lambda idx: len(docs[idx]))
    sorted_docs = [docs[idx] for idx in order]

    embedder = TextEmbedding(model_name=model_name)
    sorted_vectors: list[np.ndarray] = []
    for start in range(0, len(sorted_docs), batch_size):
        chunk = sorted_docs[start : start + batch_size]
        sorted_vectors.extend(embedder.embed(chunk, batch_size=batch_size))
        done = min(start + batch_size, len(sorted_docs))
        if done % 640 == 0 or done == len(sorted_docs):
            print(f"Encoded {done}/{len(sorted_docs)} movies", flush=True)

    matrix = np.empty((len(docs), len(sorted_vectors[0])), dtype=np.float32)
    for sorted_pos, orig_idx in enumerate(order):
        matrix[orig_idx] = sorted_vectors[sorted_pos]

    # Normalize before casting to float16 to preserve cosine direction precision
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized_f16 = (matrix / norms).astype(np.float16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, movie_ids=movie_ids, embeddings=normalized_f16)
    return output_path


if __name__ == "__main__":
    out = build_embeddings()
    print(f"Created {out} ({out.stat().st_size:,} bytes)")
