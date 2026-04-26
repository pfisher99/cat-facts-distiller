"""One-shot CatFactsGPT dataset build command."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .generate_answers import generate_answers
from .generate_questions import generate_questions
from .logging_utils import configure_logging
from .validate_dataset import validate_dataset


console = Console()


def default_thinking_out_path(out_path: str | Path) -> Path:
    output = Path(out_path)
    return output.with_name(f"{output.stem}_with_thinking{output.suffix or '.jsonl'}")


def build_dataset(
    *,
    count: int,
    out_path: str | Path,
    thinking_out_path: str | Path | None = None,
    batch_size: int = 3,
    workers: int = 4,
    avoid_context_limit: int = 250,
    question_enable_thinking: bool | None = None,
    facts_only: bool = False,
) -> tuple[int, int]:
    raw_questions = Path("data/raw/questions.jsonl")
    staged_answers = Path("data/staged/catfacts_sft_raw.jsonl")
    staged_thinking_answers = Path("data/staged/catfacts_sft_with_thinking_raw.jsonl")
    final_thinking_out = Path(thinking_out_path) if thinking_out_path else default_thinking_out_path(out_path)

    console.print("[bold]Step 1/4:[/bold] generating candidate user prompts")
    generated_questions = generate_questions(
        count=count,
        out_path=raw_questions,
        batch_size=batch_size,
        workers=workers,
        avoid_context_limit=avoid_context_limit,
        enable_thinking=question_enable_thinking,
        facts_only=facts_only,
    )

    console.print("[bold]Step 2/4:[/bold] generating assistant answers")
    generate_answers(
        in_path=raw_questions,
        out_path=staged_answers,
        thinking_out_path=staged_thinking_answers,
        workers=workers,
    )

    console.print("[bold]Step 3/4:[/bold] validating clean final JSONL")
    accepted, rejected = validate_dataset(in_path=staged_answers, out_path=out_path)

    console.print("[bold]Step 4/4:[/bold] validating thinking final JSONL")
    validate_dataset(
        in_path=staged_thinking_answers,
        out_path=final_thinking_out,
        rejected_path=Path("data/final/rejected_with_thinking.jsonl"),
        allow_thinking=True,
    )

    if generated_questions < count:
        console.print(
            f"[yellow]Only {generated_questions} prompts were generated, so the final set may be smaller than requested.[/yellow]"
        )
    return accepted, rejected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CatFactsGPT SFT dataset in one command.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True, help="Final output JSONL.")
    parser.add_argument(
        "--thinking-out",
        default=None,
        help="Final output JSONL that keeps assistant thinking. Defaults to <out>_with_thinking.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent local model requests.")
    parser.add_argument(
        "--avoid-context-limit",
        type=int,
        default=250,
        help="Existing prompts to show each question batch. Use -1 for all or 0 for none.",
    )
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
        "--facts-only",
        action="store_true",
        help="Generate only factual cat prompts and reject non-fact categories.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.verbose)
    build_dataset(
        count=args.count,
        out_path=args.out,
        thinking_out_path=args.thinking_out,
        batch_size=args.batch_size,
        workers=args.workers,
        avoid_context_limit=args.avoid_context_limit,
        question_enable_thinking=args.question_thinking,
        facts_only=args.facts_only,
    )


if __name__ == "__main__":
    main()
