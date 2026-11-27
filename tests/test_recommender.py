"""Unit and integration tests for MovieRecommender, TMDB client, FastAPI backend, and Streamlit app."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app as fastapi_app
from recommender import (
    MovieRecommender,
    build_placeholder_poster,
    fetch_movie_metadata,
    fetch_omdb_metadata,
    fetch_tmdb_metadata,
)

ROOT = Path(__file__).resolve().parent.parent


class TestMovieRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rec = MovieRecommender()
        cls.client = TestClient(fastapi_app)

    def test_deduplicated_movie_ids(self) -> None:
        ids = [int(m["movie_id"]) for m in self.rec.movies]
        self.assertEqual(len(ids), 4800)
        self.assertEqual(len(set(ids)), 4800)

    def test_offline_metadata_enriched(self) -> None:
        avatar = self.rec.get_movie(19995)
        self.assertEqual(avatar["title"], "Avatar")
        self.assertEqual(avatar["release_year"], "2009")
        self.assertGreater(avatar["rating"], 0)
        self.assertIn("Action", avatar["genres"])
        self.assertTrue(avatar["overview"])

    def test_avatar_top5_matches_expected(self) -> None:
        results = self.rec.recommend_by_title("Avatar", limit=5)
        titles = [r["title"] for r in results]
        self.assertEqual(
            titles,
            [
                "Titan A.E.",
                "Small Soldiers",
                "Independence Day",
                "Ender's Game",
                "Aliens vs Predator: Requiem",
            ],
        )

    def test_no_self_recommendation_for_previously_duplicated_movies(self) -> None:
        for mid in (268, 2661, 72710, 1255, 39269, 10844):
            results = self.rec.recommend(mid, limit=10)
            rec_ids = [r["movie_id"] for r in results]
            self.assertNotIn(mid, rec_ids)

    def test_duplicate_title_disambiguation(self) -> None:
        label_1989 = self.rec.format_movie_label(268)
        label_1966 = self.rec.format_movie_label(2661)
        self.assertNotEqual(label_1989, label_1966)
        self.assertIn("1989", label_1989)
        self.assertIn("1966", label_1966)

    def test_tmdb_missing_key_preserves_offline_fallback(self) -> None:
        avatar = self.rec.get_movie(19995)
        meta = fetch_tmdb_metadata(19995, api_key="", fallback=avatar)
        self.assertIsNone(meta.poster_url)
        self.assertEqual(meta.release_year, "2009")
        self.assertTrue(meta.overview)
        self.assertEqual(meta.tmdb_url, "https://www.themoviedb.org/movie/19995")

    def test_tmdb_timeout_and_null_poster_do_not_raise(self) -> None:
        timeout_session = MagicMock(spec=requests.Session)
        timeout_session.get.side_effect = requests.ConnectTimeout("timed out")
        meta_timeout = fetch_tmdb_metadata(19995, api_key="dummy", session=timeout_session)
        self.assertIsNone(meta_timeout.poster_url)

        null_resp = MagicMock()
        null_resp.status_code = 200
        null_resp.json.return_value = {
            "poster_path": None,
            "release_date": "2009-12-15",
            "vote_average": 7.6,
            "overview": "Test overview",
        }
        null_session = MagicMock(spec=requests.Session)
        null_session.get.return_value = null_resp
        meta_null = fetch_tmdb_metadata(19995, api_key="dummy", session=null_session)
        self.assertIsNone(meta_null.poster_url)
        self.assertEqual(meta_null.release_year, "2009")
        self.assertEqual(meta_null.rating, 7.6)
        self.assertEqual(meta_null.overview, "Test overview")

    def test_placeholder_poster_escapes_html(self) -> None:
        uri = build_placeholder_poster("Test <Movie> & Co")
        self.assertTrue(uri.startswith("data:image/svg+xml;utf8,"))
        self.assertNotIn("<Movie>", uri)

    def test_omdb_poster_and_tmdb_timeout_fallback_to_omdb(self) -> None:
        session = MagicMock(spec=requests.Session)
        omdb_resp = MagicMock()
        omdb_resp.status_code = 200
        omdb_resp.json.return_value = {
            "Response": "True",
            "Poster": "https://m.media-amazon.com/images/avatar.jpg",
            "Year": "2009",
            "imdbRating": "7.9",
            "Plot": "A marine on Pandora.",
            "imdbID": "tt0499549",
        }
        session.get.side_effect = [
            requests.ConnectTimeout("TMDB timed out"),
            omdb_resp,
        ]
        meta = fetch_movie_metadata(
            movie_id=19995,
            title="Avatar",
            release_year="2009",
            tmdb_api_key="tmdb-key",
            omdb_api_key="omdb-key",
            session=session,
        )
        self.assertEqual(meta.poster_url, "https://m.media-amazon.com/images/avatar.jpg")
        self.assertEqual(meta.rating, 7.9)
        self.assertEqual(meta.tmdb_url, "https://www.imdb.com/title/tt0499549/")

        na_session = MagicMock(spec=requests.Session)
        na_resp = MagicMock()
        na_resp.status_code = 200
        na_resp.json.return_value = {"Response": "True", "Poster": "N/A"}
        na_session.get.return_value = na_resp
        meta_na = fetch_omdb_metadata(19995, "Avatar", "2009", api_key="omdb-key", session=na_session)
        self.assertIsNone(meta_na.poster_url)

    def test_fastapi_endpoints(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["movies_count"], 4800)

        search = self.client.get("/movies", params={"q": "Avatar", "limit": 5})
        self.assertEqual(search.status_code, 200)
        self.assertTrue(any(m["movie_id"] == 19995 for m in search.json()["movies"]))

        rec = self.client.get("/recommend/19995", params={"limit": 5})
        self.assertEqual(rec.status_code, 200)
        payload = rec.json()
        self.assertEqual(payload["movie"]["title"], "Avatar")
        self.assertEqual(len(payload["recommendations"]), 5)

        missing = self.client.get("/recommend/99999999")
        self.assertEqual(missing.status_code, 404)

    def test_streamlit_app_renders_and_recommends(self) -> None:
        at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
        self.assertFalse(at.exception, f"App raised exception on load: {at.exception}")
        at.button[0].click().run(timeout=15)
        self.assertFalse(at.exception, f"App raised exception on Recommend: {at.exception}")
        self.assertGreaterEqual(len(at.subheader), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
