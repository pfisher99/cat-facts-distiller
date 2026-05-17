"""Generate candidate user prompts with the local model."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import local

import tiktoken
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from .clean import (
    ensure_parent,
    normalize_prompt,
    parse_json_from_text,
    strip_thinking_preserve_lines,
)
from .config import get_settings
from .llm_client import LocalLLMClient
from .logging_utils import configure_logging
from .prompts import (
    question_batch_prompt,
    question_generator_system_prompt,
    question_tone_directive,
)
from .schemas import (
    CATEGORIES,
    CATEGORY_ALIASES,
    DIFFICULTY_ALIASES,
    QuestionCandidate,
    QuestionRecord,
)


logger = logging.getLogger(__name__)
console = Console()
_THREAD_LOCAL = local()


@dataclass(frozen=True)
class QuestionBatchResult:
    requested: int
    items: list[QuestionCandidate]
    item_rejections: list[QuestionItemRejection]
    bad_generation: str | None = None


@dataclass(frozen=True)
class QuestionItemRejection:
    reason: str
    raw_item: object


@dataclass(frozen=True)
class QuestionFutureState:
    requested: int
    worker_id: int
    batch_index: int


def _save_bad_generation(raw: str, bad_dir: Path) -> None:
    bad_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = bad_dir / f"bad_generation_{stamp}.txt"
    path.write_text(raw, encoding="utf-8")


def _json_safe(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _question_rejection_reason(raw_item: object, exc: Exception) -> str:
    if not isinstance(raw_item, dict):
        return "not_object"

    errors_fn = getattr(exc, "errors", None)
    errors = errors_fn() if callable(errors_fn) else []
    if not errors:
        return "schema_invalid"

    for error in errors:
        loc = {str(part) for part in error.get("loc", ())}
        error_type = str(error.get("type", ""))

        if "category" in loc:
            if error_type == "missing":
                return "missing_category"
            return "invalid_category"

        if "difficulty" in loc:
            if error_type == "missing":
                return "missing_difficulty"
            return "invalid_difficulty"

        if "user_prompt" in loc:
            if error_type == "missing":
                return "missing_user_prompt"
            if error_type == "string_too_short":
                return "empty_user_prompt"
            if error_type == "string_too_long":
                return "user_prompt_too_long"
            return "invalid_user_prompt"

    return "schema_invalid"


def _coerce_question_items(payload: object) -> tuple[list[QuestionCandidate], list[QuestionItemRejection]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    elif isinstance(payload, dict) and {
        "category",
        "difficulty",
        "user_prompt",
    }.intersection(payload):
        payload = [payload]
    if not isinstance(payload, list):
        return [], [QuestionItemRejection(reason="response_not_array", raw_item=payload)]

    items: list[QuestionCandidate] = []
    rejections: list[QuestionItemRejection] = []
    for raw_item in payload:
        try:
            items.append(QuestionCandidate.model_validate(raw_item))
        except Exception as exc:  # noqa: BLE001 - generated data is expected to be messy.
            reason = _question_rejection_reason(raw_item, exc)
            rejections.append(QuestionItemRejection(reason=reason, raw_item=raw_item))
            logger.debug("Skipping malformed generated question (%s): %s", reason, exc)
    return items, rejections


def _clean_line_field(value: str) -> str:
    return value.strip().strip('"').strip("'").strip("`").strip()


QUESTION_CATEGORY_LABELS = tuple(dict.fromkeys((*CATEGORIES, *CATEGORY_ALIASES)))
QUESTION_DIFFICULTY_LABELS = tuple(dict.fromkeys(("easy", "medium", "hard", *DIFFICULTY_ALIASES)))

QUESTION_LINE_RE = re.compile(
    r"(?P<category>"
    + "|".join(re.escape(category) for category in QUESTION_CATEGORY_LABELS)
    + r")"
    r"\s*\|\s*"
    r"(?P<difficulty>"
    + "|".join(re.escape(difficulty) for difficulty in QUESTION_DIFFICULTY_LABELS)
    + r")"
    r"\s*\|\s*",
    re.IGNORECASE,
)

def _parse_question_lines(text: str) -> tuple[list[QuestionCandidate], list[QuestionItemRejection]]:
    items: list[QuestionCandidate] = []
    rejections: list[QuestionItemRejection] = []
    normalized_text = "\n".join(
        re.sub(r"^\s*(?:[-*]\s+|\d+[\).]\s*)", "", raw_line).strip()
        for raw_line in text.splitlines()
        if raw_line.strip() and not raw_line.strip().startswith("```")
    )
    matches = list(QUESTION_LINE_RE.finditer(normalized_text))

    if matches:
        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
            prompt = normalized_text[match.end() : next_start].strip()
            raw_item = {
                "category": _clean_line_field(match.group("category")),
                "difficulty": _clean_line_field(match.group("difficulty")),
                "user_prompt": _clean_line_field(prompt),
            }
            try:
                items.append(QuestionCandidate.model_validate(raw_item))
            except Exception as exc:  # noqa: BLE001 - generated data is expected to be messy.
                reason = _question_rejection_reason(raw_item, exc)
                rejections.append(QuestionItemRejection(reason=reason, raw_item=raw_item))
        return items, rejections

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^\s*(?:[-*]\s+|\d+[\).]\s*)", "", line).strip()
        if not line or line[0] in "[{":
            continue

        if "\t" in line:
            parts = line.split("\t", 2)
        else:
            parts = re.split(r"\s+\|\s+", line, maxsplit=2)

        if len(parts) != 3:
            rejections.append(QuestionItemRejection(reason="line_missing_fields", raw_item=raw_line))
            continue

        raw_item = {
            "category": _clean_line_field(parts[0]),
            "difficulty": _clean_line_field(parts[1]),
            "user_prompt": _clean_line_field(parts[2]),
        }
        try:
            items.append(QuestionCandidate.model_validate(raw_item))
        except Exception as exc:  # noqa: BLE001 - generated data is expected to be messy.
            reason = _question_rejection_reason(raw_item, exc)
            rejections.append(QuestionItemRejection(reason=reason, raw_item=raw_item))

    return items, rejections


def _extract_json_objects_from_text(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        objects.append(value)
        index = start + end
    return objects


def _parse_question_response(text: str) -> tuple[list[QuestionCandidate], list[QuestionItemRejection]]:
    stripped = text.strip()
    json_rejections: list[QuestionItemRejection] = []

    if stripped.startswith("["):
        recovered_objects = _extract_json_objects_from_text(stripped)
        if recovered_objects:
            items, object_rejections = _coerce_question_items(recovered_objects)
            if items:
                return items, object_rejections
            json_rejections = object_rejections

    if stripped.startswith(("[", "{")):
        try:
            payload = parse_json_from_text(stripped)
        except ValueError:
            payload = _extract_json_objects_from_text(stripped)

        items, json_rejections = _coerce_question_items(payload)
        if items:
            return items, json_rejections

    line_items, line_rejections = _parse_question_lines(stripped)
    if line_items:
        return line_items, line_rejections

    return [], line_rejections or json_rejections


def _append_question_rejections(
    path: Path,
    *,
    worker_id: int,
    batch_index: int,
    rejections: list[QuestionItemRejection],
) -> None:
    if not rejections:
        return
    output = ensure_parent(path)
    with output.open("a", encoding="utf-8") as handle:
        for rejection in rejections:
            handle.write(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "batch_index": batch_index + 1,
                        "reason": rejection.reason,
                        "raw_item": _json_safe(rejection.raw_item),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _thread_client(settings) -> LocalLLMClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or client.settings != settings:
        client = LocalLLMClient(settings)
        _THREAD_LOCAL.client = client
    return client


@lru_cache(maxsize=16)
def _token_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass

    for encoding_name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(encoding_name)
        except ValueError:
            continue
    raise RuntimeError("No usable tiktoken encoding is available.")


def _messages_token_count(messages: Sequence[dict[str, str]], *, model: str) -> int:
    encoding = _token_encoding(model)
    return sum(
        len(encoding.encode(message.get("role", "")))
        + len(encoding.encode(message.get("content", "")))
        + 4
        for message in messages
    ) + 2


def _question_messages(
    *,
    batch_index: int,
    requested: int,
    avoid_prompts: Sequence[str],
    facts_only: bool,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": question_generator_system_prompt(batch_index, facts_only=facts_only),
        },
        {
            "role": "user",
            "content": question_batch_prompt(
                requested,
                avoid_prompts,
                question_tone_directive(batch_index, facts_only=facts_only),
                facts_only=facts_only,
            ),
        },
    ]


def _question_messages_token_count(
    *,
    batch_index: int,
    requested: int,
    avoid_prompts: Sequence[str],
    facts_only: bool,
    model: str,
) -> int:
    return _messages_token_count(
        _question_messages(
            batch_index=batch_index,
            requested=requested,
            avoid_prompts=avoid_prompts,
            facts_only=facts_only,
        ),
        model=model,
    )


def _question_batch(
    *,
    batch_index: int,
    requested: int,
    avoid_prompts: Sequence[str],
    settings,
    enable_thinking: bool,
    facts_only: bool,
    client: LocalLLMClient | None = None,
) -> QuestionBatchResult:
    llm = client or _thread_client(settings)
    messages = _question_messages(
        batch_index=batch_index,
        requested=requested,
        avoid_prompts=avoid_prompts,
        facts_only=facts_only,
    )
    if hasattr(llm, "chat_generation"):
        try:
            generation = llm.chat_generation(messages, enable_thinking=enable_thinking)
        except TypeError:
            generation = llm.chat_generation(messages)
        raw = strip_thinking_preserve_lines(generation.raw_content or generation.clean_content)
    else:
        try:
            raw = strip_thinking_preserve_lines(llm.chat(messages, enable_thinking=enable_thinking))
        except TypeError:
            raw = strip_thinking_preserve_lines(llm.chat(messages))

    if not raw:
        return QuestionBatchResult(requested=requested, items=[], item_rejections=[])

    items, item_rejections = _parse_question_response(raw)
    bad_generation = raw if not items and raw else None
    return QuestionBatchResult(
        requested=requested,
        items=items,
        item_rejections=item_rejections,
        bad_generation=bad_generation,
    )


def _avoid_prompt_snapshot(
    prompt_history: list[str],
    limit: int,
    *,
    token_limit: int,
    batch_index: int,
    requested: int,
    facts_only: bool,
    model: str,
) -> list[str]:
    if limit == 0:
        return []
    prompts = list(prompt_history) if limit < 0 else prompt_history[-limit:]
    if token_limit <= 0 or not prompts:
        return prompts

    if (
        _question_messages_token_count(
            batch_index=batch_index,
            requested=requested,
            avoid_prompts=prompts,
            facts_only=facts_only,
            model=model,
        )
        <= token_limit
    ):
        return prompts

    low = 0
    high = len(prompts)
    while low < high:
        midpoint = (low + high) // 2
        message_tokens = _question_messages_token_count(
            batch_index=batch_index,
            requested=requested,
            avoid_prompts=prompts[midpoint:],
            facts_only=facts_only,
            model=model,
        )
        if message_tokens <= token_limit:
            high = midpoint
        else:
            low = midpoint + 1
    return prompts[low:]


def generate_questions(
    *,
    count: int,
    out_path: str | Path,
    batch_size: int = 3,
    workers: int = 4,
    avoid_context_limit: int = -1,
    avoid_context_token_limit: int | None = None,
    enable_thinking: bool | None = None,
    facts_only: bool = False,
    client: LocalLLMClient | None = None,
) -> int:
    settings = get_settings()
    question_enable_thinking = (
        settings.question_enable_thinking if enable_thinking is None else enable_thinking
    )
    question_history_token_limit = (
        settings.question_history_token_limit
        if avoid_context_token_limit is None
        else avoid_context_token_limit
    )
    output = ensure_parent(out_path)
    bad_dir = Path("data/raw/bad_generations")
    rejected_questions_path = ensure_parent("data/raw/rejected_questions.jsonl")
    rejected_questions_path.write_text("", encoding="utf-8")
    seen: set[str] = set()
    prompt_history: list[str] = []
    rows: list[dict] = []
    batch_size = max(1, batch_size)
    workers = max(1, workers)

    max_batches = max(workers, (count // batch_size + 1) * 4)
    submitted_batches = 0
    scheduled_capacity = 0
    free_worker_ids = list(range(1, workers + 1))

    def submit_next(
        executor: ThreadPoolExecutor,
        futures: dict[Future[QuestionBatchResult], QuestionFutureState],
        progress: Progress,
        worker_tasks: dict[int, TaskID],
    ) -> None:
        nonlocal submitted_batches, scheduled_capacity
        while (
            free_worker_ids
            and len(futures) < workers
            and submitted_batches < max_batches
            and len(rows) + scheduled_capacity < count
        ):
            requested = min(batch_size, count - len(rows) - scheduled_capacity)
            batch_index = submitted_batches
            worker_id = free_worker_ids.pop(0)
            avoid_prompts = _avoid_prompt_snapshot(
                prompt_history,
                avoid_context_limit,
                token_limit=question_history_token_limit,
                batch_index=batch_index,
                requested=requested,
                facts_only=facts_only,
                model=settings.model,
            )
            progress.update(
                worker_tasks[worker_id],
                description=f"worker {worker_id}: batch {batch_index + 1} asking for {requested}",
                completed=0,
                total=1,
            )
            future = executor.submit(
                _question_batch,
                batch_index=batch_index,
                requested=requested,
                avoid_prompts=avoid_prompts,
                settings=settings,
                enable_thinking=question_enable_thinking,
                facts_only=facts_only,
                client=client,
            )
            futures[future] = QuestionFutureState(
                requested=requested,
                worker_id=worker_id,
                batch_index=batch_index,
            )
            submitted_batches += 1
            scheduled_capacity += requested

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with output.open("w", encoding="utf-8") as output_handle:
        with progress:
            total_task = progress.add_task("total prompts accepted", total=count)
            worker_tasks = {
                worker_id: progress.add_task(f"worker {worker_id}: idle", total=1, completed=0)
                for worker_id in range(1, workers + 1)
            }

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: dict[Future[QuestionBatchResult], QuestionFutureState] = {}
                submit_next(executor, futures, progress, worker_tasks)

                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        state = futures.pop(future)
                        scheduled_capacity -= state.requested
                        free_worker_ids.append(state.worker_id)
                        free_worker_ids.sort()

                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - preserve long-running jobs.
                            logger.warning("Question batch failed: %s", exc)
                            progress.update(
                                worker_tasks[state.worker_id],
                                description=f"worker {state.worker_id}: batch {state.batch_index + 1} failed",
                                completed=1,
                            )
                            continue

                        if result.bad_generation:
                            _save_bad_generation(result.bad_generation, bad_dir)
                        accepted = 0
                        duplicates = 0
                        ignored = 0
                        for item in result.items:
                            if len(rows) >= count:
                                ignored += 1
                                continue
                            normalized = normalize_prompt(item.user_prompt)
                            if not normalized or normalized in seen:
                                duplicates += 1
                                continue
                            seen.add(normalized)
                            prompt_history.append(item.user_prompt)
                            question = QuestionRecord(
                                id=f"q_{len(rows) + 1:06d}",
                                category=item.category,
                                difficulty=item.difficulty,
                                user_prompt=item.user_prompt,
                            )
                            row = question.model_dump()
                            rows.append(row)
                            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output_handle.flush()
                            accepted += 1
                            progress.update(total_task, completed=min(len(rows), count))

                        all_rejections = result.item_rejections
                        _append_question_rejections(
                            rejected_questions_path,
                            worker_id=state.worker_id,
                            batch_index=state.batch_index,
                            rejections=all_rejections,
                        )

                        ignored_text = f", ignored {ignored}" if ignored else ""
                        invalid_counts = Counter(rejection.reason for rejection in all_rejections)
                        invalid_text = ""
                        if invalid_counts:
                            invalid_detail = ", ".join(
                                f"{reason}={amount}" for reason, amount in sorted(invalid_counts.items())
                            )
                            invalid_text = f", invalid {sum(invalid_counts.values())} ({invalid_detail})"
                        progress.update(
                            worker_tasks[state.worker_id],
                            description=(
                                f"worker {state.worker_id}: batch {state.batch_index + 1} "
                                f"accepted {accepted}, duplicates {duplicates}{invalid_text}{ignored_text}"
                            ),
                            completed=1,
                        )

                        if accepted == 0 and result.items:
                            logger.info("Question batch returned only duplicate prompts.")

                    submit_next(executor, futures, progress, worker_tasks)

    if len(rows) < count:
        console.print(f"[yellow]Generated {len(rows)} of {count} requested prompts.[/yellow]")
    else:
        console.print(f"[green]Wrote {len(rows)} prompts to {output}[/green]")
    if rejected_questions_path.stat().st_size:
        console.print(f"[yellow]Rejected generated prompt candidates -> {rejected_questions_path}[/yellow]")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CatFactsGPT user prompts.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent question batch requests.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--question-thinking",
        dest="question_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode for question generation. Default follows QUESTION_ENABLE_THINKING=false.",
    )
    thinking_group.add_argument(
        "--no-question-thinking",
        dest="question_thinking",
        action="store_false",
        help="Disable thinking mode for question generation.",
    )
    parser.add_argument(
        "--avoid-context-limit",
        type=int,
        default=-1,
        help="Maximum existing prompts to show each question batch. Use -1 for all or 0 for none.",
    )
    parser.add_argument(
        "--avoid-context-token-limit",
        type=int,
        default=None,
        help=(
            "Approximate max tokens for each question-agent request, including existing prompts. "
            "Defaults to QUESTION_HISTORY_TOKEN_LIMIT."
        ),
    )
    parser.add_argument(
        "--facts-only",
        action="store_true",
        help="Bias generation toward factual cat prompts without adding extra runtime category filters.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.verbose)
    generate_questions(
        count=args.count,
        out_path=args.out,
        batch_size=args.batch_size,
        workers=args.workers,
        avoid_context_limit=args.avoid_context_limit,
        avoid_context_token_limit=args.avoid_context_token_limit,
        enable_thinking=args.question_thinking,
        facts_only=args.facts_only,
    )


if __name__ == "__main__":
    main()
