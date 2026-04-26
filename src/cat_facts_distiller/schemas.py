"""Small Pydantic schemas used to keep generated rows tidy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


CATEGORIES = (
    "cat_biology",
    "cat_behavior",
    "cat_history",
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
    "cat_behavior",
    "cat_history",
    "cat_myths",
    "cat_owner_tips",
    "cat_safety",
    "weird_cat_facts",
    "short_cat_facts",
)

Category = Literal[
    "cat_biology",
    "cat_behavior",
    "cat_history",
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

    @field_validator("category", "difficulty", mode="before")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator("user_prompt")
    @classmethod
    def normalize_prompt_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class QuestionRecord(QuestionCandidate):
    id: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
