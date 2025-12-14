import unittest

from tools.memory_status import build_summary_text


class MemoryStatusTest(unittest.TestCase):
    def test_build_summary(self):
        rules = [
            {
                "scope": "critical_dialog:death",
                "text": "Move cursor",
                "usage_count": 5,
                "confidence": 0.8,
                "last_used_at": 1000,
            }
        ]
        critical = [
            {
                "timestamp": 1000,
                "scope": "critical_dialog:death",
                "delta": "hero_dead",
                "episode_id": "sample_1.json",
                "tags": ["critical_episode"],
            }
        ]
        calibrations = [
            {
                "timestamp": 1001,
                "scope": "critical_dialog:death",
                "profile": "poe_default",
                "x_norm": 0.52,
                "y_norm": 0.8,
            }
        ]
        text = build_summary_text(rules, critical, calibrations, 5, 5, 2, "critical_dialog:death", "poe_default")
        self.assertIn("Move cursor", text)
        self.assertIn("hero_dead", text)
        self.assertIn("Calibration events", text)
        self.assertIn("poe_default", text)


if __name__ == "__main__":
    unittest.main()
