import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


class ObservationActionSchemaTest(unittest.TestCase):
    def setUp(self):
        if jsonschema is None:
            self.skipTest("jsonschema not installed")
        base = Path(__file__).resolve().parents[1] / "schemas"
        self.obs_schema = json.loads((base / "observation.schema.json").read_text(encoding="utf-8"))
        self.action_schema = json.loads((base / "action.schema.json").read_text(encoding="utf-8"))

    def test_observation_sample(self):
        sample = {
            "ok": True,
            "frame_id": 120,
            "timestamp": 1712345678.1,
            "yolo_objects": [
                {"label": "enemy", "confidence": 0.82, "bbox": [0.12, 0.08, 0.26, 0.24], "extra": {}}
            ],
            "text_zones": {
                "dialog": {"text": "Play", "confidence": 0.92, "bbox": [0.12, 0.18, 0.24, 0.28]}
            },
            "player_candidate": {"label": "player", "confidence": 0.5, "bbox": [0.4, 0.2, 0.52, 0.48]},
        }
        jsonschema.validate(sample, self.obs_schema)

    def test_action_sample(self):
        sample = {
            "ok": True,
            "source": "policy_agent",
            "action": "mouse_move",
            "dx": 12,
            "dy": -3,
            "target_norm": [0.4, 0.5],
            "timestamp": 1712345678.2,
        }
        jsonschema.validate(sample, self.action_schema)

    def test_action_missing_required(self):
        sample = {"timestamp": 1712345678.2}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(sample, self.action_schema)


if __name__ == "__main__":
    unittest.main()
