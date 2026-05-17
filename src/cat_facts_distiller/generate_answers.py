"""Generate CatFactsGPT assistant answers for candidate prompts."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import local

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from .clean import clean_assistant_answer, ensure_parent
from .config import get_settings
from .llm_client import LocalLLMClient
from .logging_utils import configure_logging
from .prompts import CATFACTS_SYSTEM_PROMPT, answer_messages
from .schemas import QuestionRecord


logger = logging.getLogger(__name__)
console = Console()
_THREAD_LOCAL = local()


@dataclass(frozen=True)
class AnswerFutureState:
    index: int
    worker_id: int
    question_id: str


def _iter_questions(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield QuestionRecord.model_validate(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - skip bad generated rows.
                logger.warning("Skipping malformed question on line %s: %s", line_number, exc)


def _thread_client(settings) -> LocalLLMClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or client.settings != settings:
        client = LocalLLMClient(settings)
        _THREAD_LOCAL.client = client
    return client


def _answer_row(
    *,
    index: int,
    question: QuestionRecord,
    generated_at: str,
    settings,
    client: LocalLLMClient | None = None,
) -> tuple[int, dict | None, dict | None]:
    llm = client or _thread_client(settings)
    generation = llm.chat_generation(answer_messages(question.user_prompt))
    answer, changed_again = clean_assistant_answer(generation.clean_content)
    if not answer:
        logger.warning("Skipping %s because answer generation returned empty text.", question.id)
        return index, None, None

    base_metadata = {
        "id": question.id,
        "category": question.category,
        "difficulty": question.difficulty,
        "source_model": settings.model,
        "generated_at": generated_at,
    }
    clean_row = {
        "messages": [
            {"role": "system", "content": CATFACTS_SYSTEM_PROMPT},
            {"role": "user", "content": question.user_prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            **base_metadata,
            "cleaned": generation.cleaned or changed_again,
            "includes_thinking": False,
        },
    }

    thinking_row = None
    if generation.thinking_content:
        thinking_row = {
            "messages": [
                {"role": "system", "content": CATFACTS_SYSTEM_PROMPT},
                {"role": "user", "content": question.user_prompt},
                {"role": "assistant", "content": generation.thinking_content},
            ],
            "metadata": {
                **base_metadata,
                "cleaned": False,
                "includes_thinking": True,
                "thinking_source": generation.thinking_source,
            },
        }
    else:
        logger.info("No thinking content returned for %s; skipping thinking row.", question.id)

    return index, clean_row, thinking_row


def generate_answers(
    *,
    in_path: str | Path,
    out_path: str | Path,
    thinking_out_path: str | Path | None = None,
    workers: int = 4,
    client: LocalLLMClient | None = None,
) -> int:
    settings = get_settings()
    questions = list(_iter_questions(in_path))
    output = ensure_parent(out_path)
    thinking_output = ensure_parent(thinking_out_path) if thinking_out_path is not None else None
    generated_at = datetime.now(timezone.utc).isoformat()
    workers = max(1, workers)
    written = 0
    thinking_written = 0
    next_index = 0
    free_worker_ids = list(range(1, workers + 1))

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    def submit_next(
        executor: ThreadPoolExecutor,
        futures: dict[Future[tuple[int, dict | None, dict | None]], AnswerFutureState],
        progress: Progress,
        worker_tasks: dict[int, TaskID],
    ) -> None:
        nonlocal next_index
        while free_worker_ids and len(futures) < workers and next_index < len(questions):
            worker_id = free_worker_ids.pop(0)
            question = questions[next_index]
            progress.update(
                worker_tasks[worker_id],
                description=f"worker {worker_id}: answering {question.id}",
                completed=0,
                total=1,
            )
            future = executor.submit(
                _answer_row,
                index=next_index,
                question=question,
                generated_at=generated_at,
                settings=settings,
                client=client,
            )
            futures[future] = AnswerFutureState(
                index=next_index,
                worker_id=worker_id,
                question_id=question.id,
            )
            next_index += 1

    output_handle = output.open("w", encoding="utf-8")
    thinking_handle = thinking_output.open("w", encoding="utf-8") if thinking_output is not None else None
    try:
        with progress:
            total_task = progress.add_task("total answers generated", total=len(questions))
            worker_tasks = {
                worker_id: progress.add_task(f"worker {worker_id}: idle", total=1, completed=0)
                for worker_id in range(1, workers + 1)
            }

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: dict[Future[tuple[int, dict | None, dict | None]], AnswerFutureState] = {}
                submit_next(executor, futures, progress, worker_tasks)

                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        state = futures.pop(future)
                        free_worker_ids.append(state.worker_id)
                        free_worker_ids.sort()

                        try:
                            _index, row, thinking_row = future.result()
                        except Exception as exc:  # noqa: BLE001 - preserve long-running jobs.
                            logger.warning("Answer worker failed: %s", exc)
                            progress.update(
                                worker_tasks[state.worker_id],
                                description=f"worker {state.worker_id}: {state.question_id} failed",
                                completed=1,
                            )
                            progress.advance(total_task)
                            continue

                        if row is not None:
                            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output_handle.flush()
                            written += 1
                        if thinking_row is not None and thinking_handle is not None:
                            thinking_handle.write(json.dumps(thinking_row, ensure_ascii=False) + "\n")
                            thinking_handle.flush()
                            thinking_written += 1

                        progress.update(
                            worker_tasks[state.worker_id],
                            description=(
                                f"worker {state.worker_id}: {state.question_id} "
                                f"clean={'yes' if row else 'no'} thinking={'yes' if thinking_row else 'no'}"
                            ),
                            completed=1,
                        )
                        progress.advance(total_task)

                    submit_next(executor, futures, progress, worker_tasks)
    finally:
        output_handle.close()
        if thinking_handle is not None:
            thinking_handle.close()

    if thinking_output is not None:
        console.print(
            f"[green]Wrote {thinking_written} thinking SFT rows to {thinking_output}[/green]"
        )

    console.print(f"[green]Wrote {written} raw SFT rows to {output}[/green]")
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CatFactsGPT SFT answers.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input questions JSONL.")
    parser.add_argument("--out", required=True, help="Output staged SFT JSONL.")
    parser.add_argument(
        "--thinking-out",
        default=None,
        help="Optional staged SFT JSONL that keeps assistant thinking inside <think> tags.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent answer requests.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.verbose)
    generate_answers(
        in_path=args.input_path,
        out_path=args.out,
        thinking_out_path=args.thinking_out,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
