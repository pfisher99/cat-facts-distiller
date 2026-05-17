from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cat_facts_distiller.config import Settings  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_extra_body_includes_repetition_penalty(self) -> None:
        body = Settings(repetition_penalty=1.0).extra_body(enable_thinking=False)

        self.assertEqual(body["repetition_penalty"], 1.0)
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
