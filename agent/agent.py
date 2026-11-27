"""Hand-written tool-calling agent loop with structured output validation and hallucination guard."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.llm import LLMResponse, OpenAICompatibleLLMClient
from agent.prompts import build_system_prompt
from agent.schemas import AgentReply, Pick, ToolTraceEntry
from agent.tools import TOOL_SCHEMAS, MovieToolset

MAX_ITERATIONS = 5
DEFAULT_PER_CALL_TIMEOUT = 30.0
SAFE_FALLBACK_MESSAGE = (
    "I couldn't format structured movie picks cleanly for that request. "
    "Try naming a movie you like or describing a genre and mood (for example: 'like Inception but lighter')."
)


@dataclass
class AgentRunResult:
    """Complete output and diagnostic telemetry for a single agent turn."""

    reply: AgentReply
    trace: list[ToolTraceEntry] = field(default_factory=list)
    schema_valid_first_try: bool = False
    schema_valid_after_retry: bool = False
    used_fallback: bool = False
    hallucinated_picks_dropped: int = 0
    title_mismatches: int = 0
    total_latency_ms: float = 0.0


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _parse_agent_reply(raw_content: str | None) -> AgentReply:
    if not raw_content or not raw_content.strip():
        raise ValueError("Assistant returned empty content instead of JSON.")
    cleaned = _strip_json_fences(raw_content)
    data = json.loads(cleaned)
    return AgentReply.model_validate(data)


def _apply_hallucination_guard(
    reply: AgentReply,
    toolset: MovieToolset,
) -> tuple[AgentReply, int, int]:
    """Drop picks missing from the catalogue and verify/normalize canonical titles."""
    valid_picks: list[Pick] = []
    dropped = 0
    mismatches = 0

    for pick in reply.picks:
        try:
            canonical = toolset.recommender.get_movie(pick.movie_id)
        except KeyError:
            dropped += 1
            continue

        canonical_title = str(canonical["title"])
        if pick.title.strip().lower() != canonical_title.strip().lower():
            mismatches += 1
        valid_picks.append(
            Pick(
                movie_id=int(canonical["movie_id"]),
                title=canonical_title,
                reason=pick.reason,
            )
        )

    guarded = AgentReply.model_construct(
        picks=valid_picks,
        follow_up_question=reply.follow_up_question,
        message=reply.message,
    )
    return guarded, dropped, mismatches


def _build_safe_fallback(raw_text: str | None) -> AgentReply:
    clean_text = (raw_text or "").strip()
    if clean_text and not clean_text.startswith("{") and len(clean_text) <= 500:
        msg = clean_text
    else:
        msg = SAFE_FALLBACK_MESSAGE
    return AgentReply.model_construct(
        picks=[],
        follow_up_question="Would you like to name a favorite movie or genre to get started?",
        message=msg,
    )


class MovieAgent:
    """Iterative tool-calling agent over MovieToolset and OpenAICompatibleLLMClient."""

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient,
        toolset: MovieToolset,
        max_iterations: int = MAX_ITERATIONS,
        per_call_timeout: float = DEFAULT_PER_CALL_TIMEOUT,
    ) -> None:
        self.llm = llm_client
        self.toolset = toolset
        self.max_iterations = max(1, int(max_iterations))
        self.per_call_timeout = float(per_call_timeout)

    def run(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        preferences: dict[str, Any] | None = None,
        prompt_version: str = "v1",
        model: str | None = None,
    ) -> AgentRunResult:
        """Run the tool-calling loop and return a validated AgentReply plus trace."""
        turn_start = time.perf_counter()
        system_prompt = build_system_prompt(prompt_version=prompt_version, preferences=preferences)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in history or []:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        trace: list[ToolTraceEntry] = []
        last_response: LLMResponse | None = None

        for _ in range(self.max_iterations):
            resp = self.llm.chat_completion(
                messages=messages,
                tools=TOOL_SCHEMAS,
                timeout=self.per_call_timeout,
                model=model,
            )
            last_response = resp

            if not resp.tool_calls:
                return self._finalize_response(
                    raw_content=resp.content,
                    messages=messages,
                    trace=trace,
                    turn_start=turn_start,
                    model=model,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments_raw,
                            },
                        }
                        for call in resp.tool_calls
                    ],
                }
            )

            for call in resp.tool_calls:
                t0 = time.perf_counter()
                ok = True
                err_msg: str | None = None
                parsed_args: dict[str, Any] = {}

                try:
                    loaded = json.loads(call.arguments_raw or "{}")
                    if not isinstance(loaded, dict):
                        raise ValueError("Tool arguments must decode to a JSON object")
                    parsed_args = loaded
                    tool_result = self.toolset.execute(call.name, parsed_args)
                    if "error" in tool_result:
                        ok = False
                        err_msg = str(tool_result["error"])
                except (json.JSONDecodeError, ValueError) as exc:
                    ok = False
                    err_msg = f"Malformed tool arguments: {exc}"
                    tool_result = {"error": err_msg}

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                trace.append(
                    ToolTraceEntry(
                        name=call.name,
                        args=parsed_args,
                        latency_ms=latency_ms,
                        ok=ok,
                        error=err_msg,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self.toolset.serialize(tool_result),
                    }
                )

        # Max iterations exhausted while model kept requesting tools: request one JSON summary
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool budget reached. Using only the movie results already gathered above, "
                    "respond now with the final valid JSON object."
                ),
            }
        )
        final_resp = self.llm.chat_completion(
            messages=messages,
            tools=None,
            response_format={"type": "json_object"},
            timeout=self.per_call_timeout,
            model=model,
        )
        return self._finalize_response(
            raw_content=final_resp.content or (last_response.content if last_response else None),
            messages=messages,
            trace=trace,
            turn_start=turn_start,
            model=model,
        )

    def _finalize_response(
        self,
        raw_content: str | None,
        messages: list[dict[str, Any]],
        trace: list[ToolTraceEntry],
        turn_start: float,
        model: str | None,
    ) -> AgentRunResult:
        try:
            parsed = _parse_agent_reply(raw_content)
            guarded, dropped, mismatches = _apply_hallucination_guard(parsed, self.toolset)
            if not guarded.picks:
                fallback = _build_safe_fallback(None)
                return AgentRunResult(
                    reply=fallback,
                    trace=trace,
                    schema_valid_first_try=True,
                    used_fallback=True,
                    hallucinated_picks_dropped=dropped,
                    title_mismatches=mismatches,
                    total_latency_ms=round((time.perf_counter() - turn_start) * 1000.0, 2),
                )
            return AgentRunResult(
                reply=guarded,
                trace=trace,
                schema_valid_first_try=True,
                hallucinated_picks_dropped=dropped,
                title_mismatches=mismatches,
                total_latency_ms=round((time.perf_counter() - turn_start) * 1000.0, 2),
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            # Retry ONCE with the validation error appended
            messages.append({"role": "assistant", "content": raw_content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed JSON/schema validation: {exc}. "
                        "Reply ONLY with valid JSON matching {picks: [{movie_id, title, reason}] (1..6 items), "
                        "follow_up_question, message}. Every movie_id must come from tool results above."
                    ),
                }
            )
            try:
                retry_resp = self.llm.chat_completion(
                    messages=messages,
                    tools=None,
                    response_format={"type": "json_object"},
                    timeout=self.per_call_timeout,
                    model=model,
                )
                parsed_retry = _parse_agent_reply(retry_resp.content)
                guarded, dropped, mismatches = _apply_hallucination_guard(parsed_retry, self.toolset)
                if not guarded.picks:
                    return AgentRunResult(
                        reply=_build_safe_fallback(retry_resp.content),
                        trace=trace,
                        schema_valid_after_retry=True,
                        used_fallback=True,
                        hallucinated_picks_dropped=dropped,
                        title_mismatches=mismatches,
                        total_latency_ms=round((time.perf_counter() - turn_start) * 1000.0, 2),
                    )
                return AgentRunResult(
                    reply=guarded,
                    trace=trace,
                    schema_valid_after_retry=True,
                    hallucinated_picks_dropped=dropped,
                    title_mismatches=mismatches,
                    total_latency_ms=round((time.perf_counter() - turn_start) * 1000.0, 2),
                )
            except Exception:
                return AgentRunResult(
                    reply=_build_safe_fallback(raw_content),
                    trace=trace,
                    used_fallback=True,
                    total_latency_ms=round((time.perf_counter() - turn_start) * 1000.0, 2),
                )
