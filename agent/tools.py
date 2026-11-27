"""JSON-schema tool definitions and safe wrappers over MovieRecommender."""

from __future__ import annotations

import json
from typing import Any, Callable

from recommender import MovieRecommender
from agent.semantic import is_semantic_enabled, semantic_search_movies

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Search the 4,800-movie catalogue by movie title to find matching movie_id values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Movie title or partial title to search for (e.g. 'Inception').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (1 to 20, default 8).",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Get full metadata (year, genres, rating, runtime, short overview) for a known movie_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "Unique TMDB movie_id from the catalogue.",
                    }
                },
                "required": ["movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar",
            "description": "Find movies similar to a reference movie_id using the content-similarity index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "Source movie_id to find similar movies for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations to return (1 to 20, default 8).",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search movies by natural-language plot, mood, tone, or theme description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural-language description of desired mood, plot, or themes.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1 to 20, default 8).",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["description"],
            },
        },
    },
]


def _compact_movie(movie: dict[str, Any]) -> dict[str, Any]:
    overview = str(movie.get("overview") or "").strip()
    if len(overview) > 260:
        overview = overview[:257].rstrip() + "..."
    result: dict[str, Any] = {
        "movie_id": int(movie["movie_id"]),
        "title": str(movie["title"]),
        "release_year": str(movie.get("release_year") or ""),
        "genres": list(movie.get("genres") or []),
        "rating": movie.get("rating"),
        "overview": overview,
    }
    if "match_percent" in movie:
        result["match_percent"] = movie["match_percent"]
    if "similarity" in movie:
        result["similarity"] = movie["similarity"]
    return result


def _clamp_limit(raw_limit: Any, default: int = 8) -> int:
    if raw_limit is None:
        return default
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)):
        raise ValueError("limit must be an integer")
    return max(1, min(int(raw_limit), 20))


class MovieToolset:
    """Executes movie discovery tools safely, returning compact JSON dicts."""

    def __init__(
        self,
        recommender: MovieRecommender,
        semantic_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.recommender = recommender
        self._semantic_fn = semantic_fn

    def search_movies(self, query: Any, limit: Any = 8) -> dict[str, Any]:
        """Search movies by title (prioritizing exact and prefix matches)."""
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string"}
        try:
            k = _clamp_limit(limit, default=8)
        except ValueError as exc:
            return {"error": str(exc)}

        needle = query.strip().lower()
        exact: list[dict[str, Any]] = []
        prefix: list[dict[str, Any]] = []
        contains: list[dict[str, Any]] = []

        for movie in self.recommender.movies:
            title_lower = str(movie["title"]).lower()
            if title_lower == needle:
                exact.append(movie)
            elif title_lower.startswith(needle):
                prefix.append(movie)
            elif needle in title_lower:
                contains.append(movie)

        ordered = (exact + prefix + contains)[:k]
        return {
            "query": query.strip(),
            "count": len(ordered),
            "results": [_compact_movie(m) for m in ordered],
        }

    def get_movie_details(self, movie_id: Any) -> dict[str, Any]:
        """Return compact metadata for a single movie_id."""
        if isinstance(movie_id, bool) or not isinstance(movie_id, int):
            if isinstance(movie_id, str) and movie_id.strip().isdigit():
                movie_id = int(movie_id.strip())
            else:
                return {"error": "movie_id must be an integer"}
        try:
            movie = self.recommender.get_movie(movie_id)
        except KeyError:
            return {"error": f"Unknown movie_id: {movie_id}"}
        return _compact_movie(movie)

    def recommend_similar(self, movie_id: Any, limit: Any = 8) -> dict[str, Any]:
        """Return similar movies from the precomputed similarity matrix."""
        if isinstance(movie_id, bool) or not isinstance(movie_id, int):
            if isinstance(movie_id, str) and movie_id.strip().isdigit():
                movie_id = int(movie_id.strip())
            else:
                return {"error": "movie_id must be an integer"}
        try:
            k = _clamp_limit(limit, default=8)
            source = self.recommender.get_movie(movie_id)
            recs = self.recommender.recommend(movie_id, limit=k)
        except ValueError as exc:
            return {"error": str(exc)}
        except KeyError:
            return {"error": f"Unknown movie_id: {movie_id}"}

        return {
            "source_movie": _compact_movie(source),
            "count": len(recs),
            "results": [_compact_movie(r) for r in recs],
        }

    def semantic_search(self, description: Any, limit: Any = 8) -> dict[str, Any]:
        """Search movies by natural-language description using embeddings."""
        if not isinstance(description, str) or not description.strip():
            return {"error": "description must be a non-empty string"}
        try:
            k = _clamp_limit(limit, default=8)
        except ValueError as exc:
            return {"error": str(exc)}

        if self._semantic_fn is None and not is_semantic_enabled():
            return {"error": "Semantic search is disabled (ENABLE_SEMANTIC=0). Use search_movies or recommend_similar."}

        try:
            hits = (
                self._semantic_fn(description.strip(), k)
                if self._semantic_fn is not None
                else semantic_search_movies(description.strip(), limit=k)
            )
        except Exception as exc:
            return {"error": f"Semantic search unavailable: {exc}"}

        enriched: list[dict[str, Any]] = []
        for hit in hits[:k]:
            mid = int(hit["movie_id"])
            try:
                movie = dict(self.recommender.get_movie(mid))
            except KeyError:
                continue
            if "similarity" in hit:
                movie["similarity"] = round(float(hit["similarity"]), 4)
            enriched.append(_compact_movie(movie))

        return {
            "description": description.strip(),
            "count": len(enriched),
            "results": enriched,
        }

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name safely, never raising exceptions."""
        if not isinstance(args, dict):
            return {"error": "Tool arguments must be a JSON object"}

        try:
            if name == "search_movies":
                return self.search_movies(query=args.get("query"), limit=args.get("limit", 8))
            if name == "get_movie_details":
                return self.get_movie_details(movie_id=args.get("movie_id"))
            if name == "recommend_similar":
                return self.recommend_similar(movie_id=args.get("movie_id"), limit=args.get("limit", 8))
            if name == "semantic_search":
                return self.semantic_search(description=args.get("description"), limit=args.get("limit", 8))
            return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}

    @staticmethod
    def serialize(result: dict[str, Any]) -> str:
        """Serialize a tool result dictionary to compact JSON."""
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
