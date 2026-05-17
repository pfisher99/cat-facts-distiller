"""Small Pydantic schemas used to keep generated rows tidy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


CATEGORIES = (
    "cat_biology",
    "cat_anatomy",
    "cat_senses",
    "cat_behavior",
    "cat_communication",
    "cat_cognition",
    "cat_history",
    "cat_evolution",
    "cat_breeds",
    "cat_ecology",
    "cat_culture",
    "cat_myths",
    "cat_owner_tips",
    "cat_safety",
    "cat_jokes",
    "cat_computer_jokes",
    "weird_cat_facts",
    "short_cat_facts",
    "refusal_or_redirect",
    "stay_in_character",
)

FACT_ONLY_CATEGORIES = (
    "cat_biology",
    "cat_anatomy",
    "cat_senses",
    "cat_behavior",
    "cat_communication",
    "cat_cognition",
    "cat_history",
    "cat_evolution",
    "cat_breeds",
    "cat_ecology",
    "cat_culture",
    "cat_myths",
    "weird_cat_facts",
    "short_cat_facts",
)

CATEGORY_ALIASES = {
    "biology": "cat_biology",
    "anatomy": "cat_anatomy",
    "body": "cat_anatomy",
    "senses": "cat_senses",
    "sensory": "cat_senses",
    "behavior": "cat_behavior",
    "behavior_quirks": "cat_behavior",
    "behavioral_quirks": "cat_behavior",
    "communication": "cat_communication",
    "vocalization": "cat_communication",
    "vocalizations": "cat_communication",
    "body_language": "cat_communication",
    "cognition": "cat_cognition",
    "intelligence": "cat_cognition",
    "history": "cat_history",
    "historical_facts": "cat_history",
    "evolution": "cat_evolution",
    "domestication": "cat_evolution",
    "breeds": "cat_breeds",
    "cat_breed_facts": "cat_breeds",
    "ecology": "cat_ecology",
    "wild_cats": "cat_ecology",
    "culture": "cat_culture",
    "cat_culture_facts": "cat_culture",
    "myth_busting": "cat_myths",
    "myths": "cat_myths",
    "owner_tips": "cat_owner_tips",
    "owner_basics": "cat_owner_tips",
    "cat_hairballs": "cat_owner_tips",
    "safety_check": "cat_safety",
    "safety": "cat_safety",
    "sensory_adaptations": "cat_biology",
    "weird_facts": "weird_cat_facts",
    "weird_fact_lover": "weird_cat_facts",
    "short_facts": "short_cat_facts",
    "short_fact": "short_cat_facts",
}

DIFFICULTY_ALIASES = {
    "weird": "hard",
    "weird_fact_lover": "hard",
    "fact": "easy",
    "facts": "easy",
    "factual": "medium",
}

Category = Literal[
    "cat_biology",
    "cat_anatomy",
    "cat_senses",
    "cat_behavior",
    "cat_communication",
    "cat_cognition",
    "cat_history",
    "cat_evolution",
    "cat_breeds",
    "cat_ecology",
    "cat_culture",
    "cat_myths",
    "cat_owner_tips",
    "cat_safety",
    "cat_jokes",
    "cat_computer_jokes",
    "weird_cat_facts",
    "short_cat_facts",
    "refusal_or_redirect",
    "stay_in_character",
]

Difficulty = Literal["easy", "medium", "hard"]


class QuestionCandidate(BaseModel):
    category: Category
    difficulty: Difficulty = "easy"
    user_prompt: str = Field(min_length=1, max_length=499)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        return CATEGORY_ALIASES.get(normalized, normalized)

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        return DIFFICULTY_ALIASES.get(normalized, normalized)

    @field_validator("user_prompt")
    @classmethod
    def normalize_prompt_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class QuestionRecord(QuestionCandidate):
    id: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
