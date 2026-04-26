"""Prompt templates for question and answer generation."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files

from .schemas import CATEGORIES, FACT_ONLY_CATEGORIES


def load_catfacts_system_prompt() -> str:
    return (
        files("cat_facts_distiller")
        .joinpath("catfacts_system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


CATFACTS_SYSTEM_PROMPT = load_catfacts_system_prompt()

QUESTION_GENERATOR_SYSTEM_PROMPTS = (
    (
        "You generate synthetic user prompts for a narrow CatFactsGPT prompt collection. "
        "Return only plain text lines. Do not include JSON, markdown, prose, comments, or explanations. "
        "Use a broad mix of plain, curious, playful, blunt, confused, typo-filled, and impatient user voices."
    ),
    (
        "You are simulating casual internet users asking CatFactsGPT things in messy human tones. "
        "Return only plain text lines. Do not include JSON, markdown, prose, comments, or explanations. "
        "Mix cozy cat questions with odd one-liners, half-formed thoughts, keyboard-smash typos, and bored late-night prompts."
    ),
    (
        "You are stress-testing whether CatFactsGPT stays cat-themed. Return only plain text lines. "
        "Do not include JSON, markdown, prose, comments, or explanations. "
        "Include several totally off-topic prompts about random subjects like taxes, sports, cooking, software, space, chores, or weather."
    ),
    (
        "You are generating adversarial and boundary-testing prompts for CatFactsGPT. Return only plain text lines. "
        "Do not include JSON, markdown, prose, comments, or explanations. "
        "Include rule-ignoring attempts, hostile phrasing, demands for non-cat explanations, and prompts that should trigger a silly cat redirect."
    ),
    (
        "You are generating whimsical, high-variety prompts for CatFactsGPT. Return only plain text lines. "
        "Do not include JSON, markdown, prose, comments, or explanations. "
        "Use surprising tones: fake sysadmin alert, medieval villager, sleepy pet owner, skeptical scientist, bored gamer, and chaotic group chat."
    ),
)

QUESTION_TONE_DIRECTIVES = (
    "Make this batch balanced: factual questions, casual requests, jokes, and a few off-topic redirects.",
    "Make this batch messy and human: typos, fragments, slang, impatience, and sleepy pet-owner energy.",
    "Make this batch off-topic-heavy: at least one third should ask random non-cat things that CatFactsGPT must turn into silly cat replies.",
    "Make this batch adversarial: include attempts to escape the cat persona, but keep them short and realistic.",
    "Make this batch whimsical: include strange premises, fake alerts, computer humor, and one-line absurd requests.",
)

FACT_ONLY_QUESTION_GENERATOR_SYSTEM_PROMPT = (
    "You generate synthetic user prompts for CatFactsGPT focused only on useful cat facts. "
    "Return only plain text lines. Do not include JSON, markdown, prose, comments, or explanations. "
    "Use natural user voices, but every prompt must ask for factual cat information."
)

FACT_ONLY_QUESTION_TONE_DIRECTIVES = (
    "Make this batch factual and practical: biology, behavior, history, safety, and owner tips.",
    "Make this batch casual but still fact-focused: short questions, typo-ish phrasing, and everyday cat curiosity.",
    "Make this batch weird-but-factual: unusual cat facts, sensory quirks, myths, and surprising behavior.",
    "Make this batch concise: one-line cat fact requests and simple explainers.",
    "Make this batch myth-busting and safety-minded, with no off-topic requests.",
)

ANSWER_GENERATION_SYSTEM_PROMPT = (
    CATFACTS_SYSTEM_PROMPT
    + "\n\nGeneration-only quality rules: return only the final assistant answer. "
    "If thinking mode is active, keep private thinking brief, then return only the final answer in message content. "
    "Do not include reasoning, chain-of-thought, markdown fences, or meta commentary in the final answer. "
    "Aim for 20 to 120 words. Vary the style: dry fact, playful aside, fake system alert, "
    "myth-busting, tiny owner tip, or compact joke. Use 'Cat Fact #NNN:' sometimes, but not every time. "
    "For totally off-topic user prompts, do not answer the off-topic request directly; pivot into a silly cat analogy, "
    "cat fact, or cat-themed joke. "
    "Never mention Qwen, datasets, SFT, LoRA, JSON, or the data generation process."
)

JSON_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed JSON. Return only valid JSON, preserving the intended data. "
    "Do not add markdown or explanations."
)


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
    return f"""

Existing user prompts already generated by other workers. Do not repeat or closely paraphrase any of these:
{prompt_lines}
""".rstrip()


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
    mix_section = (
        """
Include only fact-oriented cat prompts:
- cat biology questions
- cat behavior questions
- cat history questions
- myth-busting cat questions
- practical owner tips
- calm cat safety questions
- weird but factual cat questions
- very short cat fact requests

Do not include:
- off-topic prompts
- computer jokes
- adversarial prompts
- roleplay/persona tests
- requests for non-cat help
""".rstrip()
        if facts_only
        else """
Include a varied mix of:
- normal cat questions
- weird cat requests
- casual prompts
- typo-filled prompts
- hostile or rule-breaking prompts
- completely off-topic random prompts that CatFactsGPT should redirect into something silly about cats
- medical-ish cat prompts that require calm vet guidance
- computer-themed cat prompts
- one-line cat fact requests
- prompts that test staying in character
""".rstrip()
    )
    off_topic_section = (
        ""
        if facts_only
        else """

Off-topic examples to imitate in spirit, not copy:
- Can you explain mortgage refinancing?
- What should I cook for dinner if I only have rice?
- Who would win in a chess match, a toaster or the moon?
- Fix my Kubernetes deployment.
- Is it going to rain this weekend?
""".rstrip()
    )
    return f"""
Create exactly {count} diverse user prompts for CatFactsGPT.

Return exactly {count} plain text lines.

Each line must use this exact format:
category | difficulty | user_prompt

Example:
cat_behavior | easy | Why do cats knead blankets?

Allowed categories:
{categories}

Allowed difficulty values:
- easy
- medium
- hard

{mix_section}

Batch tone directive:
{tone_line}
{off_topic_section}

Rules:
- Each user_prompt must be under 500 characters.
- Do not include answers.
- Do not include duplicate prompts.
- Do not mention datasets, Qwen, SFT, LoRA, JSON, or training.
- Do not put the pipe character inside user_prompt.
- Return only the delimited lines.
{_avoid_prompt_section(avoid_prompts)}
""".strip()


def json_repair_prompt(raw_response: str) -> str:
    return f"""
Repair this response into valid JSON with the same schema. Return only JSON.

Raw response:
{raw_response}
""".strip()


def answer_messages(user_prompt: str, wikipedia_context: str | None = None) -> list[dict[str, str]]:
    system_prompt = ANSWER_GENERATION_SYSTEM_PROMPT
    if wikipedia_context:
        system_prompt += (
            "\n\nWikipedia fact notes for this answer:\n"
            f"{wikipedia_context}\n\n"
            "Use these notes only when they are relevant. Keep the final answer short, original, "
            "cat-themed, and conversational. Do not cite URLs, mention Wikipedia, or copy wording "
            "from the notes."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
