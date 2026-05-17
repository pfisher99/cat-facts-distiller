"""Prompt templates for question and answer generation."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files

from .schemas import CATEGORIES, FACT_ONLY_CATEGORIES


def _load_prompt_text(filename: str) -> str:
    return (
        files("cat_facts_distiller")
        .joinpath(filename)
        .read_text(encoding="utf-8")
        .strip()
    )


def load_catfacts_system_prompt() -> str:
    return _load_prompt_text("catfacts_system_prompt.txt")


def _load_question_generator_system_prompts() -> tuple[str, ...]:
    return tuple(
        _load_prompt_text(f"question_generator_system_prompt_{index:02d}.txt")
        for index in range(1, 6)
    )


def _load_prompt_series(prefix: str, count: int = 5) -> tuple[str, ...]:
    return tuple(_load_prompt_text(f"{prefix}_{index:02d}.txt") for index in range(1, count + 1))


CATFACTS_SYSTEM_PROMPT = load_catfacts_system_prompt()

QUESTION_GENERATOR_SYSTEM_PROMPTS = _load_question_generator_system_prompts()

QUESTION_TONE_DIRECTIVES = _load_prompt_series("question_tone_directive")

FACT_ONLY_QUESTION_GENERATOR_SYSTEM_PROMPT = _load_prompt_text(
    "fact_only_question_generator_system_prompt.txt"
)

FACT_ONLY_QUESTION_TONE_DIRECTIVES = _load_prompt_series("fact_only_question_tone_directive")

QUESTION_BATCH_PROMPT_TEMPLATE = _load_prompt_text("question_batch_prompt_template.txt")
QUESTION_MIX_SECTION = _load_prompt_text("question_mix_section.txt")
FACT_ONLY_QUESTION_MIX_SECTION = _load_prompt_text("fact_only_question_mix_section.txt")
QUESTION_OFF_TOPIC_EXAMPLES = _load_prompt_text("question_off_topic_examples.txt")
FACT_ONLY_QUESTION_RULES = _load_prompt_text("fact_only_question_rules.txt")
QUESTION_AVOID_PROMPT_SECTION_TEMPLATE = _load_prompt_text(
    "question_avoid_prompt_section_template.txt"
)
ANSWER_GENERATION_SYSTEM_PROMPT = "\n\n".join(
    (CATFACTS_SYSTEM_PROMPT, _load_prompt_text("answer_generation_rules.txt"))
)
WIKIPEDIA_ANSWER_CONTEXT_TEMPLATE = _load_prompt_text("wikipedia_answer_context_template.txt")
JSON_REPAIR_SYSTEM_PROMPT = _load_prompt_text("json_repair_system_prompt.txt")
JSON_REPAIR_PROMPT_TEMPLATE = _load_prompt_text("json_repair_prompt_template.txt")


def question_generator_system_prompt(batch_index: int, *, facts_only: bool = False) -> str:
    if facts_only:
        return FACT_ONLY_QUESTION_GENERATOR_SYSTEM_PROMPT
    return QUESTION_GENERATOR_SYSTEM_PROMPTS[batch_index % len(QUESTION_GENERATOR_SYSTEM_PROMPTS)]


def question_tone_directive(batch_index: int, *, facts_only: bool = False) -> str:
    if facts_only:
        return FACT_ONLY_QUESTION_TONE_DIRECTIVES[
            batch_index % len(FACT_ONLY_QUESTION_TONE_DIRECTIVES)
        ]
    return QUESTION_TONE_DIRECTIVES[batch_index % len(QUESTION_TONE_DIRECTIVES)]


def _avoid_prompt_section(avoid_prompts: Sequence[str]) -> str:
    if not avoid_prompts:
        return ""
    prompt_lines = "\n".join(f"- {prompt}" for prompt in avoid_prompts)
    return QUESTION_AVOID_PROMPT_SECTION_TEMPLATE.format(prompt_lines=prompt_lines)


def question_batch_prompt(
    count: int,
    avoid_prompts: Sequence[str] = (),
    tone_directive: str = "",
    *,
    facts_only: bool = False,
) -> str:
    allowed_categories = FACT_ONLY_CATEGORIES if facts_only else CATEGORIES
    categories = "\n".join(f"- {category}" for category in allowed_categories)
    tone_line = tone_directive or question_tone_directive(0, facts_only=facts_only)
    mix_section = FACT_ONLY_QUESTION_MIX_SECTION if facts_only else QUESTION_MIX_SECTION
    off_topic_section = "" if facts_only else QUESTION_OFF_TOPIC_EXAMPLES
    fact_only_rules = FACT_ONLY_QUESTION_RULES if facts_only else ""
    return QUESTION_BATCH_PROMPT_TEMPLATE.format(
        count=count,
        categories=categories,
        mix_section=mix_section,
        tone_line=tone_line,
        off_topic_section=off_topic_section,
        fact_only_rules=fact_only_rules,
        avoid_prompt_section=_avoid_prompt_section(avoid_prompts),
    ).strip()


def json_repair_prompt(raw_response: str) -> str:
    return JSON_REPAIR_PROMPT_TEMPLATE.format(raw_response=raw_response).strip()


def answer_messages(user_prompt: str, wikipedia_context: str | None = None) -> list[dict[str, str]]:
    system_prompt = ANSWER_GENERATION_SYSTEM_PROMPT
    if wikipedia_context:
        system_prompt = "\n\n".join(
            (
                system_prompt,
                WIKIPEDIA_ANSWER_CONTEXT_TEMPLATE.format(wikipedia_context=wikipedia_context),
            )
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
