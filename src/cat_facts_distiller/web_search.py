"""Tiny optional Wikipedia lookup helper for fact-checking answer generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings
from .schemas import FACT_ONLY_CATEGORIES, QuestionRecord


logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "lookup_wikipedia"
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Look up short Wikipedia summaries about cats, cat behavior, "
            "cat health, cat safety, cat biology, or cat history before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise Wikipedia search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of Wikipedia pages to return.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

CAT_KEYWORDS = {
    "cat",
    "cats",
    "kitten",
    "kittens",
    "feline",
    "meow",
    "purr",
    "whisker",
    "litter",
    "knead",
    "vomit",
    "vet",
    "veterinarian",
}

NO_SEARCH_CATEGORIES = {"refusal_or_redirect", "stay_in_character"}


@dataclass(frozen=True)
class WikipediaResult:
    title: str
    url: str
    summary: str


@dataclass(frozen=True)
class SearchContext:
    query: str
    results: tuple[WikipediaResult, ...]
    error: str | None = None

    @property
    def attempted(self) -> bool:
        return bool(self.query)

    def prompt_context(self, max_chars: int) -> str:
        if not self.results:
            return ""
        lines = []
        for index, result in enumerate(self.results, start=1):
            source = f" ({result.url})" if result.url else ""
            lines.append(f"{index}. {result.title}{source}: {result.summary}")
        return _limit_chars("\n".join(lines), max_chars)

    def tool_content(self, max_chars: int) -> str:
        payload: dict[str, Any] = {
            "query": self.query,
            "results": [
                {"title": result.title, "url": result.url, "summary": result.summary}
                for result in self.results
            ],
        }
        if self.error:
            payload["error"] = self.error
        return _limit_chars(json.dumps(payload, ensure_ascii=False), max_chars)


def should_search(question: QuestionRecord) -> bool:
    if question.category in NO_SEARCH_CATEGORIES:
        return False
    if question.category in FACT_ONLY_CATEGORIES:
        return True
    prompt_words = set(re.findall(r"[a-zA-Z]+", question.user_prompt.lower()))
    return bool(prompt_words & CAT_KEYWORDS)


def build_search_query(question: QuestionRecord) -> str:
    prompt = re.sub(r"\s+", " ", question.user_prompt).strip()
    suffix = "cat feline fact veterinary behavior safety history"
    query = f"{prompt} {suffix}"
    return _limit_chars(query, 240)


class WikipediaLookup:
    """No-key lookup using Wikipedia summaries through MediaWiki's API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search_question(self, question: QuestionRecord) -> SearchContext | None:
        if not should_search(question):
            return None
        return self.search(build_search_query(question))

    def search(self, query: str) -> SearchContext:
        try:
            results = self._request(query)
            return SearchContext(query=query, results=tuple(results))
        except Exception as exc:  # noqa: BLE001 - lookup is optional context.
            logger.warning("Wikipedia lookup failed for %r: %s", query, exc)
            return SearchContext(query=query, results=(), error=str(exc))

    def _request(self, query: str) -> list[WikipediaResult]:
        params = urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": str(self.settings.wikipedia_results),
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "inprop": "url",
                "redirects": "1",
            }
        )
        request = Request(
            f"https://en.wikipedia.org/w/api.php?{params}",
            headers={"User-Agent": "cat-facts-distiller/0.1"},
        )
        with urlopen(request, timeout=self.settings.wikipedia_timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            return []

        results: list[WikipediaResult] = []
        for page in pages.values():
            if not isinstance(page, dict):
                continue
            title = _clean_text(page.get("title"))
            summary = _clean_text(page.get("extract"))
            if not title or not summary:
                continue
            results.append(
                WikipediaResult(
                    title=title,
                    url=_clean_text(page.get("fullurl")),
                    summary=_limit_chars(summary, 650),
                )
            )

        return _dedupe_results(results)[: self.settings.wikipedia_results]


def _dedupe_results(results: list[WikipediaResult]) -> list[WikipediaResult]:
    seen: set[tuple[str, str]] = set()
    unique: list[WikipediaResult] = []
    for result in results:
        key = (result.url.lower(), result.summary.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _clean_text(value: Any) -> str:
    if not value:
        return ""
    text = str(value).replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def _limit_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
