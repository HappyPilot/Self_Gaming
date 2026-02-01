import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import llm_client


class TestLLMClientJSONMode(unittest.TestCase):
    def setUp(self):
        llm_client.LLM_ENDPOINT = "http://example.test/v1/chat/completions"
        llm_client.LLM_MODEL = "test-model"
        llm_client.LLM_API_KEY = "test"
        llm_client._RESOLVED_MODEL = None

    def _mock_resp(self, content: str):
        class Resp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": content}}]}

        return Resp()

    def test_fetch_control_profile_uses_json_response_format(self):
        with patch("llm_client.acquire_gate", return_value=True), patch("llm_client.release_gate"), patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = self._mock_resp('{"game_id":"test"}')
            profile, status = llm_client.fetch_control_profile("hint", ["text"])
            self.assertEqual(status, "llm_ok")
            self.assertIsInstance(profile, dict)
            _, kwargs = mock_post.call_args
            body = kwargs.get("json", {})
            self.assertEqual(body.get("response_format"), {"type": "json_object"})

    def test_fetch_visual_prompts_uses_json_response_format(self):
        with patch("llm_client.acquire_gate", return_value=True), patch("llm_client.release_gate"), patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = self._mock_resp(
                '{"game_id":"test","prompts":[],"object_prompts":[],"ui_prompts":[],"confidence":0.5}'
            )
            profile, status = llm_client.fetch_visual_prompts("hint", ["text"])
            self.assertEqual(status, "llm_ok")
            self.assertIsInstance(profile, dict)
            _, kwargs = mock_post.call_args
            body = kwargs.get("json", {})
            self.assertEqual(body.get("response_format"), {"type": "json_object"})

    def test_extract_json_handles_python_style_dict(self):
        parsed = llm_client._extract_json("{'game_id': 'test_game', 'allow_mouse_move': True}")
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("game_id"), "test_game")
        self.assertTrue(parsed.get("allow_mouse_move"))

    def test_extract_json_returns_dict_when_input_is_dict(self):
        payload = {"game_id": "direct_dict", "allow_mouse_move": False}
        parsed = llm_client._extract_json(payload)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed, payload)

    def test_log_parse_failure_writes_jsonl(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            llm_client.LLM_PARSE_LOG = path
            llm_client._log_parse_failure("control_profile", "{bad json")
            with open(path, "r", encoding="utf-8") as handle:
                line = handle.readline().strip()
            data = json.loads(line)
            self.assertEqual(data.get("kind"), "control_profile")
            self.assertIn("raw", data)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
