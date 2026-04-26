"""Runtime configuration for local OpenAI-compatible generation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    openai_base_url: str = "http://127.0.0.1:8000/v1"
    openai_api_key: str = "EMPTY"
    model: str = "Qwen3.5-9B-local"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 8192
    presence_penalty: float = 1.5
    enable_thinking: bool = True
    question_enable_thinking: bool = False
    enable_wikipedia_lookup: bool = False
    wikipedia_results: int = 3
    wikipedia_timeout: float = 8.0
    wikipedia_max_context_chars: int = 1800
    request_retries: int = 3
    request_timeout: float = 120.0

    def extra_body(self, *, enable_thinking: bool | None = None) -> dict:
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        body: dict = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        return body


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_base_url=os.getenv("OPENAI_BASE_URL", Settings.openai_base_url),
        openai_api_key=os.getenv("OPENAI_API_KEY", Settings.openai_api_key),
        model=os.getenv("OPENAI_MODEL", Settings.model),
        temperature=_float_env("TEMPERATURE", Settings.temperature),
        top_p=_float_env("TOP_P", Settings.top_p),
        top_k=_int_env("TOP_K", Settings.top_k),
        min_p=_float_env("MIN_P", Settings.min_p),
        max_tokens=_int_env("MAX_TOKENS", Settings.max_tokens),
        presence_penalty=_float_env("PRESENCE_PENALTY", Settings.presence_penalty),
        enable_thinking=_bool_env("ENABLE_THINKING", Settings.enable_thinking),
        question_enable_thinking=_bool_env(
            "QUESTION_ENABLE_THINKING",
            Settings.question_enable_thinking,
        ),
        enable_wikipedia_lookup=_bool_env(
            "ENABLE_WIKIPEDIA_LOOKUP",
            _bool_env("ENABLE_WEB_SEARCH", Settings.enable_wikipedia_lookup),
        ),
        wikipedia_results=_int_env("WIKIPEDIA_RESULTS", Settings.wikipedia_results),
        wikipedia_timeout=_float_env("WIKIPEDIA_TIMEOUT", Settings.wikipedia_timeout),
        wikipedia_max_context_chars=_int_env(
            "WIKIPEDIA_MAX_CONTEXT_CHARS",
            Settings.wikipedia_max_context_chars,
        ),
        request_retries=_int_env("REQUEST_RETRIES", Settings.request_retries),
        request_timeout=_float_env("REQUEST_TIMEOUT", Settings.request_timeout),
    )
