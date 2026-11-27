"""Core movie recommendation engine and resilient TMDB/OMDb metadata clients."""

from __future__ import annotations

import gzip
import html
import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_MOVIES_PATH = DATA_DIR / "movies_clean.json.gz"
DEFAULT_SIMILARITY_PATH = DATA_DIR / "similarity_topk.npz"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
OMDB_API_BASE = "https://www.omdbapi.com/"

_HTTP_SESSION = requests.Session()


@dataclass(frozen=True)
class MovieMetadata:
    """Metadata fetched from TMDB/OMDb merged with offline dataset defaults."""

    movie_id: int
    poster_url: str | None = None
    release_year: str = ""
    rating: float | None = None
    runtime: int | None = None
    genres: tuple[str, ...] = ()
    tagline: str = ""
    overview: str = ""
    tmdb_url: str = ""


def build_placeholder_poster(title: str) -> str:
    """Return an inline SVG data URI poster when no external poster is available."""
    safe_title = html.escape(title[:48])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" viewBox="0 0 500 750">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#1e293b"/><stop offset="100%" stop-color="#0f172a"/>'
        "</linearGradient></defs>"
        '<rect width="500" height="750" fill="url(#g)" rx="16"/>'
        '<rect x="24" y="24" width="452" height="702" fill="none" stroke="#334155" stroke-width="2" rx="12"/>'
        '<text x="250" y="320" text-anchor="middle" fill="#64748b" font-family="sans-serif" '
        'font-size="22" font-weight="600">POSTER UNAVAILABLE</text>'
        f'<text x="250" y="385" text-anchor="middle" fill="#e2e8f0" font-family="sans-serif" '
        f'font-size="26" font-weight="700">{safe_title}</text>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def _fallback_metadata(movie_id: int, fallback: dict[str, Any] | None) -> MovieMetadata:
    fb = fallback or {}
    return MovieMetadata(
        movie_id=movie_id,
        release_year=str(fb.get("release_year") or ""),
        rating=fb.get("rating"),
        runtime=fb.get("runtime"),
        genres=tuple(str(g) for g in (fb.get("genres") or [])),
        tagline=str(fb.get("tagline") or ""),
        overview=str(fb.get("overview") or ""),
        tmdb_url=f"https://www.themoviedb.org/movie/{movie_id}",
    )


def fetch_tmdb_metadata(
    movie_id: int,
    api_key: str | None = None,
    timeout: float = 4.0,
    session: requests.Session | None = None,
    fallback: dict[str, Any] | None = None,
) -> MovieMetadata:
    """Fetch movie poster and details from TMDB without raising on network/API errors."""
    base = _fallback_metadata(movie_id, fallback)
    key = (api_key if api_key is not None else os.getenv("TMDB_API_KEY", "")).strip()
    if not key:
        return base

    http = session or _HTTP_SESSION
    url = f"{TMDB_API_BASE}/movie/{movie_id}"
    headers: dict[str, str] = {"Accept": "application/json"}
    params: dict[str, str] = {"language": "en-US"}

    # Support both TMDB v4 JWT Bearer tokens and v3 API keys
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    else:
        params["api_key"] = key

    try:
        response = http.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return base
        data: dict[str, Any] = response.json()
    except (requests.RequestException, ValueError):
        return base

    poster_path = data.get("poster_path")
    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    release_date = str(data.get("release_date") or "")
    release_year = release_date[:4] if len(release_date) >= 4 else base.release_year
    vote_avg = data.get("vote_average")
    rating = round(float(vote_avg), 1) if isinstance(vote_avg, (int, float)) and vote_avg > 0 else base.rating
    overview = str(data.get("overview") or "").strip() or base.overview
    tagline = str(data.get("tagline") or "").strip() or base.tagline
    runtime_val = data.get("runtime")
    runtime = int(runtime_val) if isinstance(runtime_val, (int, float)) and runtime_val > 0 else base.runtime
    genres_raw = data.get("genres")
    genres = (
        tuple(str(g["name"]) for g in genres_raw if isinstance(g, dict) and "name" in g)
        if isinstance(genres_raw, list) and genres_raw
        else base.genres
    )

    return MovieMetadata(
        movie_id=movie_id,
        poster_url=poster_url,
        release_year=release_year,
        rating=rating,
        runtime=runtime,
        genres=genres,
        tagline=tagline,
        overview=overview,
        tmdb_url=base.tmdb_url,
    )


def fetch_omdb_metadata(
    movie_id: int,
    title: str,
    release_year: str = "",
    api_key: str | None = None,
    timeout: float = 4.0,
    session: requests.Session | None = None,
    fallback: dict[str, Any] | None = None,
) -> MovieMetadata:
    """Fetch movie poster and details from OMDb without raising on network/API errors."""
    base = _fallback_metadata(movie_id, fallback)
    key = (api_key if api_key is not None else os.getenv("OMDB_API_KEY", "")).strip()
    if not key or not title.strip():
        return base

    http = session or _HTTP_SESSION

    def _query(with_year: bool) -> dict[str, Any] | None:
        params: dict[str, str] = {"t": title.strip(), "type": "movie", "apikey": key}
        if with_year and release_year:
            params["y"] = release_year
        try:
            resp = http.get(OMDB_API_BASE, params=params, timeout=timeout)
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("Response") == "True":
                return payload
        except (requests.RequestException, ValueError):
            return None
        return None

    data = _query(with_year=True)
    if data is None and release_year:
        data = _query(with_year=False)
    if data is None:
        return base

    raw_poster = str(data.get("Poster") or "").strip()
    poster_url = raw_poster if raw_poster.startswith("http") else None
    raw_year = str(data.get("Year") or "").strip()[:4]
    year = raw_year if raw_year.isdigit() else base.release_year

    rating = base.rating
    try:
        imdb_rating = float(data.get("imdbRating", ""))
        if imdb_rating > 0:
            rating = round(imdb_rating, 1)
    except (TypeError, ValueError):
        pass

    plot = str(data.get("Plot") or "").strip()
    overview = plot if plot and plot != "N/A" else base.overview

    imdb_id = str(data.get("imdbID") or "").strip()
    info_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else base.tmdb_url

    return MovieMetadata(
        movie_id=movie_id,
        poster_url=poster_url,
        release_year=year,
        rating=rating,
        runtime=base.runtime,
        genres=base.genres,
        tagline=base.tagline,
        overview=overview,
        tmdb_url=info_url,
    )


def fetch_movie_metadata(
    movie_id: int,
    title: str = "",
    release_year: str = "",
    tmdb_api_key: str | None = None,
    omdb_api_key: str | None = None,
    timeout: float = 4.0,
    session: requests.Session | None = None,
    fallback: dict[str, Any] | None = None,
) -> MovieMetadata:
    """Resolve metadata using OMDb and/or TMDB, with automatic fallback."""
    omdb_key = (omdb_api_key if omdb_api_key is not None else os.getenv("OMDB_API_KEY", "")).strip()
    tmdb_key = (tmdb_api_key if tmdb_api_key is not None else os.getenv("TMDB_API_KEY", "")).strip()

    # Try TMDB if configured; fall back to OMDb if TMDB yields no poster (or is blocked)
    if tmdb_key:
        tmdb_meta = fetch_tmdb_metadata(
            movie_id=movie_id,
            api_key=tmdb_key,
            timeout=timeout,
            session=session,
            fallback=fallback,
        )
        if tmdb_meta.poster_url or not omdb_key:
            return tmdb_meta

    if omdb_key:
        effective_title = title or str((fallback or {}).get("title") or "")
        effective_year = release_year or str((fallback or {}).get("release_year") or "")
        return fetch_omdb_metadata(
            movie_id=movie_id,
            title=effective_title,
            release_year=effective_year,
            api_key=omdb_key,
            timeout=timeout,
            session=session,
            fallback=fallback,
        )

    return _fallback_metadata(movie_id, fallback)


class MovieRecommender:
    """Fast content-based movie recommender backed by precomputed top-K similarities."""

    def __init__(
        self,
        movies_path: Path = DEFAULT_MOVIES_PATH,
        similarity_path: Path = DEFAULT_SIMILARITY_PATH,
    ) -> None:
        if not movies_path.exists() or not similarity_path.exists():
            raise FileNotFoundError(
                f"Missing recommendation artifacts ({movies_path}, {similarity_path}). "
                "Run `python scripts/build_artifacts.py` to generate them."
            )

        with gzip.open(movies_path, "rt", encoding="utf-8") as gz:
            self.movies: list[dict[str, Any]] = json.load(gz)

        with np.load(similarity_path) as data:
            self.top_indices: np.ndarray = data["indices"]
            self.top_scores: np.ndarray = data["scores"]

        self._id_to_row: dict[int, int] = {
            int(movie["movie_id"]): idx for idx, movie in enumerate(self.movies)
        }

        title_counts: dict[str, int] = {}
        for movie in self.movies:
            title = str(movie["title"])
            title_counts[title] = title_counts.get(title, 0) + 1
        self._dup_titles: set[str] = {t for t, count in title_counts.items() if count > 1}

    @property
    def max_k(self) -> int:
        return int(self.top_indices.shape[1])

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        """Return the movie record for `movie_id`."""
        mid = int(movie_id)
        if mid not in self._id_to_row:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        return self.movies[self._id_to_row[mid]]

    def format_movie_label(self, movie_id: int) -> str:
        """Return a unique human-readable label, disambiguating duplicate titles."""
        movie = self.get_movie(movie_id)
        title = str(movie["title"])
        if title in self._dup_titles:
            year = movie.get("release_year") or f"TMDB #{movie_id}"
            return f"{title} ({year})"
        return title

    def list_movies(self, query: str = "", limit: int | None = None) -> list[dict[str, Any]]:
        """Return movies optionally filtered by case-insensitive title or genre substring."""
        q = query.strip().lower()
        records: list[dict[str, Any]] = []
        for movie in self.movies:
            title = str(movie["title"])
            genres = movie.get("genres") or []
            if q and q not in title.lower() and not any(q in str(g).lower() for g in genres):
                continue
            mid = int(movie["movie_id"])
            records.append(
                {
                    "movie_id": mid,
                    "title": title,
                    "label": self.format_movie_label(mid),
                    "release_year": movie.get("release_year", ""),
                    "rating": movie.get("rating"),
                    "genres": genres,
                }
            )
            if limit is not None and len(records) >= limit:
                break
        return records

    def recommend(self, movie_id: int, limit: int = 5) -> list[dict[str, Any]]:
        """Return top `limit` similar movies for the given `movie_id`."""
        mid = int(movie_id)
        if mid not in self._id_to_row:
            raise KeyError(f"Unknown movie_id: {movie_id}")

        clamped_limit = max(1, min(int(limit), self.max_k))
        row_idx = self._id_to_row[mid]
        neighbor_rows = self.top_indices[row_idx][:clamped_limit]
        neighbor_scores = self.top_scores[row_idx][:clamped_limit]

        results: list[dict[str, Any]] = []
        for n_row, score in zip(neighbor_rows.tolist(), neighbor_scores.tolist()):
            rec = self.movies[n_row]
            results.append(
                {
                    "movie_id": int(rec["movie_id"]),
                    "title": str(rec["title"]),
                    "score": float(score),
                    "match_percent": round(float(score) * 100, 1),
                    "release_year": rec.get("release_year", ""),
                    "rating": rec.get("rating"),
                    "runtime": rec.get("runtime"),
                    "genres": rec.get("genres", []),
                    "tagline": rec.get("tagline", ""),
                    "overview": rec.get("overview", ""),
                }
            )
        return results

    def recommend_by_title(self, title: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return top `limit` similar movies matching the first exact title."""
        for movie in self.movies:
            if movie["title"] == title:
                return self.recommend(int(movie["movie_id"]), limit=limit)
        raise KeyError(f"Unknown movie title: {title!r}")
