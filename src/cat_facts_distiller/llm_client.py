"""Thin OpenAI-compatible client wrapper for local Qwen generation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

from openai import OpenAI

from .clean import (
    clean_assistant_answer,
    compose_thinking_answer,
    has_thinking,
    normalize_thinking_message,
    parse_json_from_text,
    strip_think_tags,
)
from .config import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatGeneration:
    clean_content: str
    raw_content: str
    reasoning_content: str
    thinking_content: str | None
    thinking_source: str | None
    cleaned: bool


class LocalLLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.request_timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        enable_thinking: bool | None = None,
        raise_on_failure: bool = False,
    ) -> str:
        generation = self.chat_generation(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            raise_on_failure=raise_on_failure,
        )
        return generation.clean_content

    def chat_generation(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        enable_thinking: bool | None = None,
        raise_on_failure: bool = False,
    ) -> ChatGeneration:
        last_error: Exception | None = None
        attempts = max(1, self.settings.request_retries)

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    temperature=self.settings.temperature if temperature is None else temperature,
                    top_p=self.settings.top_p if top_p is None else top_p,
                    presence_penalty=(
                        self.settings.presence_penalty
                        if presence_penalty is None
                        else presence_penalty
                    ),
                    max_tokens=self.settings.max_tokens if max_tokens is None else max_tokens,
                    extra_body=self.settings.extra_body(enable_thinking=enable_thinking),
                )
                return self._extract_generation(response)
            except Exception as exc:  # noqa: BLE001 - keep long dataset runs alive.
                last_error = exc
                logger.warning("LLM request failed on attempt %s/%s: %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))

        if raise_on_failure and last_error is not None:
            raise last_error
        return ChatGeneration(
            clean_content="",
            raw_content="",
            reasoning_content="",
            thinking_content=None,
            thinking_source=None,
            cleaned=False,
        )

    def request_json(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool | None = None,
    ) -> tuple[Any | None, str]:
        raw = self.chat(messages, enable_thinking=enable_thinking)
        if not raw:
            return None, raw
        try:
            return parse_json_from_text(raw), raw
        except ValueError as exc:
            logger.warning("Could not parse JSON response: %s", exc)
            return None, raw

    @staticmethod
    def _message_field(message: Any, field: str) -> Any:
        if isinstance(message, dict):
            return message.get(field)

        value = getattr(message, field, None)
        if value is not None:
            return value

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(field)

        return None

    @classmethod
    def _extract_generation(cls, response: Any) -> ChatGeneration:
        choice = response.choices[0]
        message = choice.message

        reasoning_source = "reasoning_content"
        reasoning_content = cls._message_field(message, "reasoning_content")
        if not reasoning_content:
            reasoning_content = cls._message_field(message, "reasoning")
            reasoning_source = "reasoning"
        reasoning_content = reasoning_content or ""
        reasoning_content = strip_think_tags(str(reasoning_content))

        raw_content = cls._message_field(message, "content") or ""
        raw_content = str(raw_content).replace("\x00", "").strip()
        clean_content, cleaned = clean_assistant_answer(raw_content)

        thinking_content: str | None = None
        thinking_source: str | None = None
        if has_thinking(raw_content):
            thinking_content = normalize_thinking_message(raw_content)
            thinking_source = "content_tags"
        elif reasoning_content and clean_content:
            thinking_content = compose_thinking_answer(reasoning_content, clean_content)
            thinking_source = reasoning_source

        return ChatGeneration(
            clean_content=clean_content,
            raw_content=raw_content,
            reasoning_content=reasoning_content,
            thinking_content=thinking_content,
            thinking_source=thinking_source,
            cleaned=cleaned,
        )
