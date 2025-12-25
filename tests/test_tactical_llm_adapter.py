import os
import unittest

from tactical import llm_adapter


class TacticalLLMAdapterTest(unittest.TestCase):
    def test_parse_update_from_text_embedded_json(self):
        text = "here is json ... {\"global_strategy\": {\"mode\": \"scan\"}, \"targets\": {}, \"cooldowns\": {}} ... thanks"
        parsed = llm_adapter._parse_update_from_text(text)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("global_strategy", {}).get("mode"), "scan")

    def test_extract_content_from_choices(self):
        payload = {"choices": [{"message": {"content": "{\"ok\": true}"}}]}
        content = llm_adapter._extract_content(payload)
        self.assertEqual(content, "{\"ok\": true}")

    def test_extract_content_from_message(self):
        payload = {"message": {"content": "hello"}}
        content = llm_adapter._extract_content(payload)
        self.assertEqual(content, "hello")

    def test_normalize_update_defaults(self):
        old_mode = os.environ.get("TACTICAL_DEFAULT_MODE")
        os.environ["TACTICAL_DEFAULT_MODE"] = "scan"
        try:
            update = llm_adapter._normalize_update(None)
        finally:
            if old_mode is None:
                os.environ.pop("TACTICAL_DEFAULT_MODE", None)
            else:
                os.environ["TACTICAL_DEFAULT_MODE"] = old_mode
        self.assertEqual(update["global_strategy"]["mode"], "scan")
        self.assertIn("ts", update["global_strategy"])
        self.assertEqual(update["targets"], {})
        self.assertEqual(update["cooldowns"], {})


if __name__ == "__main__":
    unittest.main()
