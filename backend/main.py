"""FastAPI backend for Render deployment, sharing core logic with recommender.py."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from recommender import (
    MovieMetadata,
    MovieRecommender,
    fetch_movie_metadata,
)

app = FastAPI(
    title="Movie Recommender API",
    version="1.1.0",
    description="Content-based movie recommendation API across 4,800 movies.",
)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
allowed_origins = (
    ["*"]
    if raw_origins == "*" or not raw_origins
    else [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    """Load the compact recommender artifacts once per process."""
    return MovieRecommender()


@lru_cache(maxsize=4096)
def _cached_metadata_lookup(movie_id: int, tmdb_key: str, omdb_key: str) -> MovieMetadata:
    rec = get_recommender()
    fallback = rec.get_movie(movie_id)
    return fetch_movie_metadata(
        movie_id=movie_id,
        title=str(fallback.get("title") or ""),
        release_year=str(fallback.get("release_year") or ""),
        tmdb_api_key=tmdb_key,
        omdb_api_key=omdb_key,
        fallback=fallback,
    )


def _enrich_movie(movie: dict[str, Any], meta: MovieMetadata) -> dict[str, Any]:
    return {
        "movie_id": int(movie["movie_id"]),
        "title": str(movie["title"]),
        "label": get_recommender().format_movie_label(int(movie["movie_id"])),
        "score": movie.get("score"),
        "match_percent": movie.get("match_percent"),
        "poster_url": meta.poster_url,
        "release_year": meta.release_year,
        "rating": meta.rating,
        "runtime": meta.runtime,
        "genres": list(meta.genres),
        "tagline": meta.tagline,
        "overview": meta.overview,
        "tmdb_url": meta.tmdb_url,
    }


def _fetch_metadata_batch(movie_ids: list[int]) -> dict[int, MovieMetadata]:
    tmdb_key = os.getenv("TMDB_API_KEY", "").strip()
    omdb_key = os.getenv("OMDB_API_KEY", "").strip()
    if not tmdb_key and not omdb_key:
        rec = get_recommender()
        return {
            mid: fetch_movie_metadata(
                movie_id=mid, tmdb_api_key="", omdb_api_key="", fallback=rec.get_movie(mid)
            )
            for mid in movie_ids
        }
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(movie_ids)))) as pool:
        results = list(
            pool.map(lambda mid: _cached_metadata_lookup(mid, tmdb_key, omdb_key), movie_ids)
        )
    return {meta.movie_id: meta for meta in results}


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness and readiness check for Render."""
    rec = get_recommender()
    return {
        "status": "ok",
        "movies_count": len(rec.movies),
        "omdb_configured": bool(os.getenv("OMDB_API_KEY", "").strip()),
        "tmdb_configured": bool(os.getenv("TMDB_API_KEY", "").strip()),
    }


@app.get("/movies")
def list_movies(
    q: str = Query(default="", description="Optional substring filter on title or genre"),
    limit: int = Query(default=30, ge=1, le=5000),
) -> dict[str, Any]:
    """Search or list available movies."""
    rec = get_recommender()
    items = rec.list_movies(query=q, limit=limit)
    return {"count": len(items), "movies": items}


@app.get("/movie/{movie_id}")
def get_movie_details(movie_id: int) -> dict[str, Any]:
    """Return details for a single movie."""
    rec = get_recommender()
    try:
        movie = rec.get_movie(movie_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found") from exc

    meta = _fetch_metadata_batch([movie_id])[movie_id]
    return _enrich_movie(movie, meta)


@app.get("/recommend/{movie_id}")
def recommend_movies(
    movie_id: int,
    limit: int = Query(default=10, ge=1, le=25),
) -> dict[str, Any]:
    """Return the selected movie and top-K similar recommendations."""
    rec = get_recommender()
    try:
        source_movie = rec.get_movie(movie_id)
        recs = rec.recommend(movie_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found") from exc

    all_ids = [movie_id] + [int(r["movie_id"]) for r in recs]
    meta_by_id = _fetch_metadata_batch(all_ids)

    return {
        "movie": _enrich_movie(source_movie, meta_by_id[movie_id]),
        "recommendations": [_enrich_movie(r, meta_by_id[int(r["movie_id"])]) for r in recs],
    }
