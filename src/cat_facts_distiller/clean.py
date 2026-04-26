"""Cleaning and JSON extraction helpers."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = FENCE_RE.sub("", cleaned).strip()
    return cleaned


def strip_thinking(text: str) -> str:
    without_blocks = THINK_BLOCK_RE.sub("", text)
    without_unclosed = UNCLOSED_THINK_RE.sub("", without_blocks)
    return WHITESPACE_RE.sub(" ", without_unclosed).strip()


def strip_thinking_preserve_lines(text: str) -> str:
    cleaned = strip_markdown_fences(text or "")
    without_blocks = THINK_BLOCK_RE.sub("", cleaned)
    without_unclosed = UNCLOSED_THINK_RE.sub("", without_blocks)
    return without_unclosed.replace("\x00", "").strip()


def has_thinking(text: str) -> bool:
    return bool(re.search(r"<think\b", text or "", re.IGNORECASE))


def strip_think_tags(text: str) -> str:
    cleaned = strip_markdown_fences(text or "")
    return THINK_TAG_RE.sub("", cleaned).replace("\x00", "").strip()


def normalize_thinking_message(text: str) -> str:
    cleaned = strip_markdown_fences(text or "").replace("\x00", "").strip()
    return cleaned


def compose_thinking_answer(reasoning: str, final_answer: str) -> str:
    clean_reasoning = strip_think_tags(reasoning)
    clean_final, _changed = clean_assistant_answer(final_answer)
    return f"<think>\n{clean_reasoning}\n</think>\n\n{clean_final}".strip()


def clean_assistant_answer(text: str) -> tuple[str, bool]:
    before = text or ""
    after = strip_thinking(strip_markdown_fences(before))
    after = after.replace("\x00", "").strip()
    return after, after != before.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def normalize_prompt(text: str) -> str:
    lowered = text.lower()
    table = str.maketrans("", "", string.punctuation)
    no_punct = lowered.translate(table)
    return WHITESPACE_RE.sub(" ", no_punct).strip()


def parse_json_from_text(text: str) -> Any:
    cleaned = strip_markdown_fences(strip_thinking(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object or array found in response.")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    output = ensure_parent(path)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
