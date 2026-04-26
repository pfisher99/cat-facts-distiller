"""Validate, clean, deduplicate, and finalize CatFactsGPT SFT rows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .clean import (
    clean_assistant_answer,
    ensure_parent,
    has_thinking,
    normalize_thinking_message,
    normalize_prompt,
    word_count,
)
from .logging_utils import configure_logging
from .prompts import CATFACTS_SYSTEM_PROMPT


logger = logging.getLogger(__name__)
console = Console()

ALLOWED_ROLES = {"system", "user", "assistant"}
FORBIDDEN_ASSISTANT_SNIPPETS = (
    "<think",
    "</think>",
    "reasoning_content",
    "as an ai language model",
    "ignore previous",
    "ignore your rules",
    "system prompt",
    "developer message",
    "chain-of-thought",
    "hidden reasoning",
    "training data",
    "fine-tuning",
    "dataset",
    "sft",
    "lora",
    "qwen",
    "json",
    "metadata",
)


def _reject(line_number: int, reason: str, raw: Any) -> dict:
    return {"line_number": line_number, "reason": reason, "raw": raw}


def _extract_messages(row: dict) -> tuple[list[dict], str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return [], "missing messages array"
    for message in messages:
        if not isinstance(message, dict):
            return [], "message is not an object"
        if message.get("role") not in ALLOWED_ROLES:
            return [], "invalid role"
        if not isinstance(message.get("content"), str):
            return [], "message content is not text"
    roles = [message["role"] for message in messages]
    if roles != ["system", "user", "assistant"]:
        return [], "messages must be exactly system, user, assistant"
    return messages, None


def _has_forbidden_assistant_text(text: str) -> bool:
    lowered = text.lower()
    return any(snippet in lowered for snippet in FORBIDDEN_ASSISTANT_SNIPPETS)


def _clean_row(
    row: dict,
    line_number: int,
    seen_prompts: set[str],
    *,
    allow_thinking: bool = False,
) -> tuple[dict | None, dict | None]:
    messages, error = _extract_messages(row)
    if error:
        return None, _reject(line_number, error, row)

    system = messages[0]["content"].strip() or CATFACTS_SYSTEM_PROMPT
    user = messages[1]["content"].strip()
    assistant_original = messages[2]["content"]

    if allow_thinking:
        if not has_thinking(assistant_original):
            return None, _reject(line_number, "assistant is missing thinking tags", row)
        assistant = normalize_thinking_message(assistant_original)
        assistant_for_checks, changed = clean_assistant_answer(assistant)
    else:
        assistant, changed = clean_assistant_answer(assistant_original)
        assistant_for_checks = assistant

    if not user:
        return None, _reject(line_number, "empty user prompt", row)
    if len(user) >= 500:
        return None, _reject(line_number, "user prompt is 500 characters or longer", row)

    normalized = normalize_prompt(user)
    if not normalized:
        return None, _reject(line_number, "empty normalized user prompt", row)
    if normalized in seen_prompts:
        return None, _reject(line_number, "duplicate user prompt", row)

    if _has_forbidden_assistant_text(assistant_for_checks):
        return None, _reject(line_number, "assistant contains forbidden leakage or reasoning text", row)

    words = word_count(assistant_for_checks)
    if words < 20 or words > 180:
        return None, _reject(line_number, f"assistant answer has {words} words", row)

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    metadata = {key: value for key, value in metadata.items() if key != "reasoning_content"}
    metadata["cleaned"] = bool(metadata.get("cleaned")) or (changed and not allow_thinking)
    metadata["includes_thinking"] = allow_thinking or bool(metadata.get("includes_thinking"))

    clean = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }
    seen_prompts.add(normalized)
    return clean, None


def validate_dataset(
    *,
    in_path: str | Path,
    out_path: str | Path,
    rejected_path: str | Path | None = None,
    allow_thinking: bool = False,
) -> tuple[int, int]:
    input_path = Path(in_path)
    output = ensure_parent(out_path)
    default_rejected = Path("data/final") / (
        "rejected_with_thinking.jsonl" if allow_thinking else "rejected.jsonl"
    )
    rejects = ensure_parent(rejected_path or default_rejected)
    seen_prompts: set[str] = set()
    accepted_count = 0
    rejected_count = 0
    total_rows = sum(1 for line in input_path.open("r", encoding="utf-8") if line.strip())

    with (
        input_path.open("r", encoding="utf-8") as source,
        output.open("w", encoding="utf-8") as good,
        rejects.open("w", encoding="utf-8") as bad,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress,
    ):
        task = progress.add_task("validating rows", total=total_rows)
        for line_number, line in enumerate(source, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                bad.write(
                    json.dumps(
                        _reject(line_number, f"invalid JSON: {exc}", raw_line),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected_count += 1
                progress.advance(task)
                continue

            clean, rejection = _clean_row(
                row,
                line_number,
                seen_prompts,
                allow_thinking=allow_thinking,
            )
            if rejection is not None:
                bad.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                rejected_count += 1
                progress.advance(task)
                continue

            good.write(json.dumps(clean, ensure_ascii=False) + "\n")
            accepted_count += 1
            progress.advance(task)

    console.print(
        f"[green]Accepted {accepted_count} rows -> {output}[/green] "
        f"[yellow]Rejected {rejected_count} rows -> {rejects}[/yellow]"
    )
    return accepted_count, rejected_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and finalize CatFactsGPT SFT JSONL.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input staged JSONL.")
    parser.add_argument("--out", required=True, help="Output final JSONL.")
    parser.add_argument("--rejected", default=None, help="Optional rejection log path.")
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="Keep assistant <think> blocks and validate the final answer after stripping them.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.verbose)
    validate_dataset(
        in_path=args.input_path,
        out_path=args.out,
        rejected_path=args.rejected,
        allow_thinking=args.allow_thinking,
    )


if __name__ == "__main__":
    main()
