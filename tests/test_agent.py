"""Offline unit tests for tools, LLM client, agent loop, schema retry, and hallucination guard."""

from __future__ import annotations

import json
import unittest
from typing import Any

import httpx

from agent.agent import MovieAgent
from agent.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    LLMToolCall,
    OpenAICompatibleLLMClient,
)
from agent.tools import MovieToolset
from recommender import MovieRecommender


class FakeLLMClient:
    """Deterministic scripted LLM client for offline agent testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class TestAgentCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommender = MovieRecommender()
        cls.toolset = MovieToolset(cls.recommender)

    def test_tools_dispatch_and_error_recovery(self) -> None:
        search_res = self.toolset.execute("search_movies", {"query": "Inception", "limit": 3})
        self.assertEqual(search_res["results"][0]["movie_id"], 27205)

        details = self.toolset.execute("get_movie_details", {"movie_id": 27205})
        self.assertEqual(details["title"], "Inception")

        sim = self.toolset.execute("recommend_similar", {"movie_id": 27205, "limit": 3})
        self.assertEqual(len(sim["results"]), 3)

        bad_id = self.toolset.execute("get_movie_details", {"movie_id": 99999999})
        self.assertIn("error", bad_id)

        bad_tool = self.toolset.execute("unknown_tool", {})
        self.assertIn("error", bad_tool)

        bad_query = self.toolset.execute("search_movies", {"query": ""})
        self.assertIn("error", bad_query)

    def test_llm_client_retries_429_and_rejects_missing_key(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] < 2:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [{"message": {"content": "ok", "tool_calls": []}}],
                },
            )

        client = OpenAICompatibleLLMClient(
            api_key="test-key",
            backoff_seconds=0.0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        resp = client.chat_completion(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(resp.content, "ok")
        self.assertEqual(attempts["count"], 2)

        def always_429(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "still limited"})

        client_fail = OpenAICompatibleLLMClient(
            api_key="test-key",
            max_retries=1,
            backoff_seconds=0.0,
            http_client=httpx.Client(transport=httpx.MockTransport(always_429)),
        )
        with self.assertRaises(LLMRateLimitError):
            client_fail.chat_completion(messages=[{"role": "user", "content": "hi"}])

        no_key = OpenAICompatibleLLMClient(api_key="")
        with self.assertRaises(LLMAuthError):
            no_key.chat_completion(messages=[{"role": "user", "content": "hi"}])

    def test_agent_tool_loop_and_malformed_args_recovery(self) -> None:
        llm = FakeLLMClient(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        LLMToolCall(id="c1", name="search_movies", arguments_raw="{bad json"),
                        LLMToolCall(
                            id="c2",
                            name="search_movies",
                            arguments_raw='{"query": "Inception", "limit": 3}',
                        ),
                    ],
                ),
                LLMResponse(
                    content=json.dumps(
                        {
                            "picks": [
                                {
                                    "movie_id": 27205,
                                    "title": "Inception",
                                    "reason": "Mind-bending heist thriller.",
                                }
                            ],
                            "follow_up_question": "Want something lighter?",
                            "message": "Here is a great match.",
                        }
                    )
                ),
            ]
        )
        agent = MovieAgent(llm_client=llm, toolset=self.toolset)
        result = agent.run("Something like Inception")

        self.assertTrue(result.schema_valid_first_try)
        self.assertEqual(len(result.trace), 2)
        self.assertFalse(result.trace[0].ok)
        self.assertTrue(result.trace[1].ok)
        self.assertEqual(result.reply.picks[0].movie_id, 27205)

    def test_agent_max_iterations_cap(self) -> None:
        infinite_tool_call = LLMResponse(
            content=None,
            tool_calls=[
                LLMToolCall(id="loop", name="search_movies", arguments_raw='{"query": "Avatar"}')
            ],
        )
        final_summary = LLMResponse(
            content=json.dumps(
                {
                    "picks": [
                        {
                            "movie_id": 19995,
                            "title": "Avatar",
                            "reason": "Epic sci-fi adventure.",
                        }
                    ],
                    "follow_up_question": None,
                    "message": "Here is Avatar.",
                }
            )
        )
        llm = FakeLLMClient([infinite_tool_call, infinite_tool_call, infinite_tool_call, final_summary])
        agent = MovieAgent(llm_client=llm, toolset=self.toolset, max_iterations=3)
        result = agent.run("Avatar")

        self.assertEqual(len(result.trace), 3)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(result.reply.picks[0].movie_id, 19995)

    def test_agent_schema_retry_and_fallback(self) -> None:
        llm_retry = FakeLLMClient(
            [
                LLMResponse(content='{"picks": [], "message": "missing picks"}'),
                LLMResponse(
                    content=json.dumps(
                        {
                            "picks": [
                                {
                                    "movie_id": 19995,
                                    "title": "Avatar",
                                    "reason": "Rich alien world.",
                                }
                            ],
                            "follow_up_question": None,
                            "message": "Fixed JSON response.",
                        }
                    )
                ),
            ]
        )
        agent_retry = MovieAgent(llm_client=llm_retry, toolset=self.toolset)
        res_retry = agent_retry.run("Give me a sci-fi epic")
        self.assertFalse(res_retry.schema_valid_first_try)
        self.assertTrue(res_retry.schema_valid_after_retry)
        self.assertFalse(res_retry.used_fallback)
        self.assertEqual(len(llm_retry.calls), 2)

        llm_fail = FakeLLMClient(
            [
                LLMResponse(content="Here is some plain text without JSON."),
                LLMResponse(content="Still not valid JSON!"),
            ]
        )
        agent_fail = MovieAgent(llm_client=llm_fail, toolset=self.toolset)
        res_fail = agent_fail.run("hi")
        self.assertTrue(res_fail.used_fallback)
        self.assertEqual(res_fail.reply.picks, [])

    def test_hallucination_guard_drops_fake_id_and_normalizes_title(self) -> None:
        llm = FakeLLMClient(
            [
                LLMResponse(
                    content=json.dumps(
                        {
                            "picks": [
                                {
                                    "movie_id": 99999999,
                                    "title": "Fake Movie Not In Catalogue",
                                    "reason": "Hallucinated.",
                                },
                                {
                                    "movie_id": 27205,
                                    "title": "Inception (2010)",
                                    "reason": "Real movie with slightly off title.",
                                },
                            ],
                            "follow_up_question": None,
                            "message": "Filtered picks.",
                        }
                    )
                )
            ]
        )
        agent = MovieAgent(llm_client=llm, toolset=self.toolset)
        res = agent.run("Mind-bending movies")

        self.assertEqual(res.hallucinated_picks_dropped, 1)
        self.assertEqual(res.title_mismatches, 1)
        self.assertEqual(len(res.reply.picks), 1)
        self.assertEqual(res.reply.picks[0].movie_id, 27205)
        self.assertEqual(res.reply.picks[0].title, "Inception")


if __name__ == "__main__":
    unittest.main(verbosity=2)
