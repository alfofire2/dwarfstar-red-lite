import json
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ApiReferenceTests(unittest.TestCase):
    def test_every_prompt_has_a_reference_answer(self):
        prompts = [line for line in (FIXTURES / "api_prompts.txt").read_text().splitlines()
                   if line and not line.startswith("#")]
        answers = json.loads((FIXTURES / "qwen_api_reference.json").read_text())["answers"]
        self.assertEqual(len(prompts), len(set(prompts)), "duplicate prompt")
        self.assertEqual(sorted(prompts), sorted(answers),
                         "run python3 scripts/dev/api_compare.py fetch after editing api_prompts.txt")


if __name__ == "__main__":
    unittest.main()
