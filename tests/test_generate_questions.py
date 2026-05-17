from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cat_facts_distiller.generate_questions import generate_questions  # noqa: E402


class FakeQuestionClient:
    def __init__(self, responses: list[str], output_path: Path | None = None) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []
        self.output_path = output_path
        self.output_snapshots: list[str] = []

    def chat(self, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
        self.calls.append(messages)
        if self.output_path is not None:
            if self.output_path.exists():
                self.output_snapshots.append(self.output_path.read_text(encoding="utf-8"))
            else:
                self.output_snapshots.append("")
        return self.responses.pop(0)


class GenerateQuestionsTests(unittest.TestCase):
    def test_facts_only_accepts_valid_mixed_categories_and_keeps_basic_guards(self) -> None:
        response = json.dumps(
            [
                {
                    "category": "cat_owner_tips",
                    "difficulty": "easy",
                    "user_prompt": "Should I rotate cat toys?",
                },
                {
                    "category": "cat_safety",
                    "difficulty": "medium",
                    "user_prompt": "Is chocolate dangerous for cats?",
                },
                {
                    "category": "cat_owner_tips",
                    "difficulty": "hard",
                    "user_prompt": "Should I rotate cat toys?",
                },
                {
                    "category": "not_a_category",
                    "difficulty": "easy",
                    "user_prompt": "This one should still be schema-rejected.",
                },
                {
                    "category": "refusal_or_redirect",
                    "difficulty": "easy",
                    "user_prompt": "Explain taxes but make it cats.",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                output_path = Path(tmpdir) / "questions.jsonl"
                generated = generate_questions(
                    count=3,
                    out_path=output_path,
                    batch_size=5,
                    workers=1,
                    avoid_context_limit=0,
                    facts_only=True,
                    client=FakeQuestionClient([response]),
                )

                rows = [
                    json.loads(line)
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                ]
                rejections = [
                    json.loads(line)
                    for line in Path("data/raw/rejected_questions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(generated, 3)
        self.assertEqual(
            [row["category"] for row in rows],
            ["cat_owner_tips", "cat_safety", "refusal_or_redirect"],
        )
        self.assertEqual(
            [row["user_prompt"] for row in rows],
            [
                "Should I rotate cat toys?",
                "Is chocolate dangerous for cats?",
                "Explain taxes but make it cats.",
            ],
        )
        self.assertEqual([rejection["reason"] for rejection in rejections], ["invalid_category"])

    def test_questions_are_written_as_they_are_accepted(self) -> None:
        responses = [
            "cat_behavior | easy | Why do cats knead blankets?",
            "cat_senses | easy | Why do cats like sunbeams?",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                output_path = Path(tmpdir) / "questions.jsonl"
                client = FakeQuestionClient(responses, output_path=output_path)
                generated = generate_questions(
                    count=2,
                    out_path=output_path,
                    batch_size=1,
                    workers=1,
                    avoid_context_limit=0,
                    client=client,
                )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(generated, 2)
        self.assertEqual(len(client.output_snapshots), 2)
        self.assertEqual(client.output_snapshots[0], "")
        self.assertIn("Why do cats knead blankets?", client.output_snapshots[1])
        self.assertNotIn("Why do cats like sunbeams?", client.output_snapshots[1])


if __name__ == "__main__":
    unittest.main()
