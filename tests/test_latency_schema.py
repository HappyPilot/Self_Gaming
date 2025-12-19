import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


class LatencySchemaTest(unittest.TestCase):
    def setUp(self):
        if jsonschema is None:
            self.skipTest("jsonschema not installed")
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "latency_event.schema.json"
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def test_valid_sample(self):
        sample = {
            "event": "latency",
            "stage": "detect",
            "duration_ms": 42.3,
            "sla_ms": 100,
            "timestamp": 1712345678.9,
            "tags": {"agent": "object_detection_agent", "frame_id": 120},
        }
        jsonschema.validate(sample, self.schema)

    def test_missing_required(self):
        sample = {
            "event": "latency",
            "duration_ms": 42.3,
            "timestamp": 1712345678.9,
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(sample, self.schema)


if __name__ == "__main__":
    unittest.main()
