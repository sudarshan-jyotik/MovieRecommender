"""Pydantic schemas for the conversational movie agent and API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class Pick(BaseModel):
    """A single validated movie recommendation chosen by the agent."""

    movie_id: int
    title: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class AgentReply(BaseModel):
    """Structured final response produced by the conversational agent."""

    picks: list[Pick] = Field(..., min_length=1, max_length=6)
    follow_up_question: str | None = None
    message: str = Field(..., min_length=1)


class ToolTraceEntry(BaseModel):
    """Execution telemetry for a single tool invocation."""

    name: str
    args: dict[str, Any]
    latency_ms: float
    ok: bool = True
    error: str | None = None


class PreferenceUpdate(BaseModel):
    """Structured user preference delta extracted from a conversation turn."""

    liked_genres: list[str] = Field(default_factory=list)
    disliked_genres: list[str] = Field(default_factory=list)
    liked_movies: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Incoming payload for POST /chat."""

    session_id: str | None = Field(default=None, max_length=128)
    message: str = Field(..., min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace")
        return stripped


class ChatResponse(BaseModel):
    """Response payload for POST /chat."""

    session_id: str
    reply: AgentReply
    trace: list[ToolTraceEntry] = Field(default_factory=list)
    preferences: dict[str, list[str]] = Field(default_factory=dict)
