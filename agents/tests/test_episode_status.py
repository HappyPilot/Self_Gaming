import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.episode_status import scan_episode_dir


class EpisodeStatusTest(unittest.TestCase):
    def test_scan_returns_counts_and_last_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            first = base / "sample_1_00001.json"
            first.write_text(json.dumps({"timestamp": 1, "stage": 0, "delta": "hero_dead"}), encoding="utf-8")
            time.sleep(0.01)
            second = base / "sample_2_00002.json"
            second.write_text(json.dumps({"timestamp": 2, "stage": 1, "delta": "hero_resurrected"}), encoding="utf-8")
            summary = scan_episode_dir(base)
            self.assertEqual(summary.count, 2)
            self.assertEqual(summary.last_stage, 1)
            self.assertEqual(summary.last_delta, "hero_resurrected")
            self.assertEqual(summary.last_file, second.name)


if __name__ == "__main__":
    unittest.main()
