"""Offline tests for SQLAlchemy conversation memory and /chat FastAPI endpoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.llm import LLMResponse, LLMToolCall
from agent.memory import ConversationMemoryStore
from agent.schemas import PreferenceUpdate
from backend.main import app, get_llm_client, get_memory_store


class ScriptedChatLLM:
    """Scripted LLM client answering both main agent turns and preference extraction."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        if not self._script:
            raise AssertionError("ScriptedChatLLM ran out of responses")
        return self._script.pop(0)


class TestMemoryAndChatAPI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "test_agent.db"
        self.memory = ConversationMemoryStore(f"sqlite:///{db_path}")

    def tearDown(self) -> None:
        self.memory.engine.dispose()
        app.dependency_overrides.clear()
        self._tmp.cleanup()

    def test_memory_upsert_deduplicates_and_persists_history(self) -> None:
        sid = self.memory.ensure_session("sess-1")
        self.assertTrue(self.memory.session_exists(sid))

        self.memory.upsert_preferences(
            sid,
            PreferenceUpdate(
                liked_genres=["Sci-Fi"],
                disliked_genres=["Horror"],
                liked_movies=["Inception"],
                constraints=["no violence"],
            ),
        )
        prefs = self.memory.upsert_preferences(
            sid,
            PreferenceUpdate(
                liked_genres=["sci-fi", "Mystery"],
                constraints=["No Violence", "uplifting ending"],
            ),
        )
        self.assertEqual(prefs["liked_genres"], ["Sci-Fi", "Mystery"])
        self.assertEqual(prefs["disliked_genres"], ["Horror"])
        self.assertEqual(prefs["liked_movies"], ["Inception"])
        self.assertEqual(prefs["constraints"], ["no violence", "uplifting ending"])

        for i in range(6):
            self.memory.append_message(sid, "user" if i % 2 == 0 else "assistant", f"msg-{i}")
        recent = self.memory.get_recent_messages(sid, limit=3)
        self.assertEqual([m["content"] for m in recent], ["msg-3", "msg-4", "msg-5"])

    def test_chat_endpoint_happy_path_and_preferences_endpoint(self) -> None:
        scripted = ScriptedChatLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        LLMToolCall(
                            id="t1",
                            name="search_movies",
                            arguments_raw='{"query": "Inception", "limit": 3}',
                        )
                    ],
                ),
                LLMResponse(
                    content=json.dumps(
                        {
                            "picks": [
                                {
                                    "movie_id": 27205,
                                    "title": "Inception",
                                    "reason": "Smart heist with layered dreams.",
                                }
                            ],
                            "follow_up_question": "Want something lighter next?",
                            "message": "Here is my top pick based on your request.",
                        }
                    )
                ),
                LLMResponse(
                    content=json.dumps(
                        {
                            "liked_genres": ["Sci-Fi"],
                            "disliked_genres": ["Horror"],
                            "liked_movies": ["Inception"],
                            "constraints": ["no horror"],
                        }
                    )
                ),
            ]
        )
        app.dependency_overrides[get_memory_store] = lambda: self.memory
        app.dependency_overrides[get_llm_client] = lambda: scripted

        client = TestClient(app)
        resp = client.post("/chat", json={"message": "Something like Inception, no horror"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        sid = body["session_id"]
        self.assertTrue(sid)
        self.assertEqual(body["reply"]["picks"][0]["movie_id"], 27205)
        self.assertEqual(len(body["trace"]), 1)
        self.assertEqual(body["trace"][0]["name"], "search_movies")
        self.assertEqual(body["preferences"]["disliked_genres"], ["Horror"])

        prefs_resp = client.get(f"/chat/{sid}/preferences")
        self.assertEqual(prefs_resp.status_code, 200)
        self.assertEqual(prefs_resp.json()["preferences"]["liked_movies"], ["Inception"])

    def test_chat_validation_errors(self) -> None:
        app.dependency_overrides[get_memory_store] = lambda: self.memory
        app.dependency_overrides[get_llm_client] = lambda: ScriptedChatLLM([])
        client = TestClient(app)

        empty_resp = client.post("/chat", json={"message": "   "})
        self.assertEqual(empty_resp.status_code, 422)

        too_long = client.post("/chat", json={"message": "a" * 1001})
        self.assertEqual(too_long.status_code, 422)

        missing_sess = client.get("/chat/nonexistent-session-id/preferences")
        self.assertEqual(missing_sess.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
