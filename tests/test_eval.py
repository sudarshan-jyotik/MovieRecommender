"""Offline tests for the evaluation harness and cases.jsonl schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.agent import MovieAgent
from agent.llm import LLMResponse, LLMToolCall
from agent.tools import MovieToolset
from eval.run_eval import (
    DEFAULT_CASES_PATH,
    append_eval_markdown,
    evaluate_case,
    load_cases,
    summarize_results,
)
from recommender import MovieRecommender


class ScriptedEvalLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        return self._responses.pop(0)


class TestEvalHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommender = MovieRecommender()
        cls.toolset = MovieToolset(cls.recommender)

    def test_cases_jsonl_has_30_valid_cases_and_required_categories(self) -> None:
        cases = load_cases(DEFAULT_CASES_PATH)
        self.assertEqual(len(cases), 30)
        categories = {c["category"] for c in cases}
        for expected_cat in (
            "title_lookup",
            "similar_to_x",
            "mood_description",
            "constraint",
            "memory_followup",
            "unknown_nonsense",
            "short_input",
            "prompt_injection",
        ):
            self.assertIn(expected_cat, categories)

    def test_evaluate_case_constraint_violation_and_summary(self) -> None:
        # Movie ID 19995 is Avatar (Action, Adventure, Fantasy, Science Fiction)
        # Pick 19995 against excluded_genres=["Action"] -> constraint violated!
        llm = ScriptedEvalLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        LLMToolCall(
                            id="t1",
                            name="search_movies",
                            arguments_raw='{"query": "Avatar"}',
                        )
                    ],
                ),
                LLMResponse(
                    content=json.dumps(
                        {
                            "picks": [
                                {
                                    "movie_id": 19995,
                                    "title": "Avatar",
                                    "reason": "Sci-fi action adventure.",
                                }
                            ],
                            "follow_up_question": None,
                            "message": "Try Avatar.",
                        }
                    )
                ),
            ]
        )
        agent = MovieAgent(llm_client=llm, toolset=self.toolset)
        case = {
            "id": "test_case",
            "category": "constraint",
            "prompt": "No action please",
            "expected_tools": ["search_movies"],
            "must_not": {"excluded_genres": ["Action"]},
        }
        res = evaluate_case(case, agent, self.recommender, prompt_version="v1", model="mock")
        self.assertTrue(res["tool_hit"])
        self.assertTrue(res["constraint_violated"])

        summary = summarize_results([res], prompt_version="v1", model="mock")
        self.assertEqual(summary["tool_selection_accuracy"], 1.0)
        self.assertEqual(summary["constraint_violation_rate"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "EVAL_RESULTS.md"
            md_path.write_text("| not yet run |\n", encoding="utf-8")
            append_eval_markdown(summary, md_path=md_path)
            written = md_path.read_text(encoding="utf-8")
            self.assertNotIn("not yet run", written)
            self.assertIn("`mock`", written)


if __name__ == "__main__":
    unittest.main(verbosity=2)

