"""Provider-agnostic OpenAI-compatible chat-completions client over httpx."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2


class LLMError(RuntimeError):
    """Base exception for LLM client failures."""


class LLMAuthError(LLMError):
    """Raised when LLM API credentials are missing or rejected."""


class LLMRateLimitError(LLMError):
    """Raised when rate limits persist after retries."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


@dataclass(frozen=True)
class LLMToolCall:
    """Normalized tool invocation requested by the LLM."""

    id: str
    name: str
    arguments_raw: str


@dataclass(frozen=True)
class LLMResponse:
    """Normalized chat-completion response."""

    content: str | None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleLLMClient:
    """Thin httpx client for OpenAI-compatible `/chat/completions` endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = 0.75,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("LLM_API_KEY", "")).strip()
        raw_base = (base_url if base_url is not None else os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).strip()
        self.base_url = (raw_base or DEFAULT_BASE_URL).rstrip("/")
        raw_model = (model if model is not None else os.getenv("LLM_MODEL", DEFAULT_MODEL)).strip()
        self.model = raw_model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = http_client

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with retry/backoff on 429 and 5xx."""
        if not self.api_key:
            raise LLMAuthError("LLM_API_KEY is not configured.")

        effective_model = (model or self.model).strip()
        effective_timeout = timeout if timeout is not None else self.timeout
        endpoint = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            try:
                response = self._post(endpoint, headers=headers, payload=payload, timeout=effective_timeout)
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("LLM request timed out (attempt %s/%s, model=%s)", attempt + 1, attempts, effective_model)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise LLMTimeoutError(f"LLM request timed out after {effective_timeout}s") from exc
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning("LLM transport error (attempt %s/%s, model=%s)", attempt + 1, attempts, effective_model)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise LLMError(f"LLM transport error: {exc.__class__.__name__}") from exc

            status = response.status_code
            logger.info("LLM response status=%s model=%s attempt=%s", status, effective_model, attempt + 1)

            if status in (401, 403):
                raise LLMAuthError(f"LLM authentication failed with HTTP {status}")

            if status == 429 or status >= 500:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                if status == 429:
                    raise LLMRateLimitError("LLM rate limit exceeded after retries")
                raise LLMError(f"LLM server error HTTP {status} after retries")

            if status >= 400:
                raise LLMError(f"LLM request rejected with HTTP {status}: {response.text[:240]}")

            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError("LLM returned invalid JSON payload") from exc

            return self._parse_response(data, fallback_model=effective_model)

        raise LLMError(f"LLM call failed: {last_exc}")

    def _post(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        if self._client is not None:
            return self._client.post(endpoint, headers=headers, json=payload, timeout=timeout)
        with httpx.Client(timeout=timeout) as client:
            return client.post(endpoint, headers=headers, json=payload)

    @staticmethod
    def _parse_response(data: dict[str, Any], fallback_model: str) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("LLM response contained no choices")
        message = choices[0].get("message") or {}
        raw_calls = message.get("tool_calls") or []
        tool_calls: list[LLMToolCall] = []
        for idx, call in enumerate(raw_calls):
            fn = call.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            tool_calls.append(
                LLMToolCall(
                    id=str(call.get("id") or f"call_{idx}"),
                    name=str(fn.get("name") or ""),
                    arguments_raw=raw_args,
                )
            )
        content = message.get("content")
        return LLMResponse(
            content=str(content) if content is not None else None,
            tool_calls=tool_calls,
            model=str(data.get("model") or fallback_model),
            usage=dict(data.get("usage") or {}),
        )
