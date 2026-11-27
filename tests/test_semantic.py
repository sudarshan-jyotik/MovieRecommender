"""Offline unit tests for semantic search and ENABLE_SEMANTIC flag handling."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.semantic import SemanticUnavailableError, semantic_search_movies
from agent.tools import MovieToolset
from recommender import MovieRecommender


class FakeEmbedder:
    """Deterministic unit-vector embedder for offline tests."""

    def embed(self, texts: list[str]):
        for _ in texts:
            yield np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestSemanticSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommender = MovieRecommender()

    def test_semantic_search_cosine_topk_and_tool_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "movie_embeddings.npz"
            movie_ids = np.asarray([19995, 27205, 155], dtype=np.int32)
            embeddings = np.asarray(
                [
                    [0.2, 0.9, 0.0, 0.0],
                    [0.98, 0.1, 0.0, 0.0],
                    [-0.5, 0.5, 0.0, 0.0],
                ],
                dtype=np.float16,
            )
            np.savez_compressed(npz_path, movie_ids=movie_ids, embeddings=embeddings)

            hits = semantic_search_movies(
                "mind-bending dream heist",
                limit=2,
                embeddings_path=npz_path,
                embedder=FakeEmbedder(),
            )
            self.assertEqual([h["movie_id"] for h in hits], [27205, 19995])

            toolset = MovieToolset(
                self.recommender,
                semantic_fn=lambda desc, lim: semantic_search_movies(
                    desc, limit=lim, embeddings_path=npz_path, embedder=FakeEmbedder()
                ),
            )
            res = toolset.execute("semantic_search", {"description": "dream heist", "limit": 2})
            self.assertEqual(res["count"], 2)
            self.assertEqual(res["results"][0]["title"], "Inception")

    def test_semantic_disabled_returns_tool_error(self) -> None:
        prev = os.environ.get("ENABLE_SEMANTIC")
        os.environ["ENABLE_SEMANTIC"] = "0"
        try:
            with self.assertRaises(SemanticUnavailableError):
                semantic_search_movies("space adventure", embedder=FakeEmbedder())
            toolset = MovieToolset(self.recommender)
            res = toolset.execute("semantic_search", {"description": "space adventure"})
            self.assertIn("error", res)
        finally:
            if prev is None:
                os.environ.pop("ENABLE_SEMANTIC", None)
            else:
                os.environ["ENABLE_SEMANTIC"] = prev


if __name__ == "__main__":
    unittest.main(verbosity=2)

