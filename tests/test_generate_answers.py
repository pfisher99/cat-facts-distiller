from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cat_facts_distiller.generate_answers import generate_answers  # noqa: E402


class FakeAnswerClient:
    def __init__(self, responses: list[str], output_path: Path, thinking_path: Path) -> None:
        self.responses = responses
        self.output_path = output_path
        self.thinking_path = thinking_path
        self.output_snapshots: list[str] = []
        self.thinking_snapshots: list[str] = []

    def chat_generation(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        if self.output_path.exists():
            self.output_snapshots.append(self.output_path.read_text(encoding="utf-8"))
        else:
            self.output_snapshots.append("")
        if self.thinking_path.exists():
            self.thinking_snapshots.append(self.thinking_path.read_text(encoding="utf-8"))
        else:
            self.thinking_snapshots.append("")

        answer = self.responses.pop(0)
        return SimpleNamespace(
            clean_content=answer,
            cleaned=False,
            thinking_content=f"<think>\nthinking about {answer}\n</think>\n\n{answer}",
            thinking_source="reasoning_content",
        )


class GenerateAnswersTests(unittest.TestCase):
    def test_answers_are_written_as_they_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "questions.jsonl"
            output_path = tmp_path / "answers.jsonl"
            thinking_path = tmp_path / "answers_with_thinking.jsonl"
            input_rows = [
                {
                    "id": "q_000001",
                    "category": "cat_behavior",
                    "difficulty": "easy",
                    "user_prompt": "Why do cats knead blankets?",
                },
                {
                    "id": "q_000002",
                    "category": "cat_senses",
                    "difficulty": "easy",
                    "user_prompt": "Why do cats like sunbeams?",
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in input_rows),
                encoding="utf-8",
            )

            client = FakeAnswerClient(
                ["Cats knead because kitten habits stick.", "Sunbeams are warm cat magnets."],
                output_path,
                thinking_path,
            )
            written = generate_answers(
                in_path=input_path,
                out_path=output_path,
                thinking_out_path=thinking_path,
                workers=1,
                client=client,
            )

        self.assertEqual(written, 2)
        self.assertEqual(len(client.output_snapshots), 2)
        self.assertEqual(client.output_snapshots[0], "")
        self.assertIn("Cats knead because kitten habits stick.", client.output_snapshots[1])
        self.assertNotIn("Sunbeams are warm cat magnets.", client.output_snapshots[1])
        self.assertIn("Cats knead because kitten habits stick.", client.thinking_snapshots[1])


if __name__ == "__main__":
    unittest.main()
