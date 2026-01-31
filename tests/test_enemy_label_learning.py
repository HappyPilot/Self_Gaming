import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "agents"))

from scene_agent import _update_dynamic_enemy_labels


class EnemyLabelLearningTest(unittest.TestCase):
    def test_update_dynamic_enemy_labels(self):
        labels = set(["enemy"])
        counts = {}
        objects = [{"label": "goblin", "bbox": [0.1, 0.1, 0.2, 0.2]}]
        bars = [{"bbox": [0.1, 0.1, 0.2, 0.2]}]
        updated, counts = _update_dynamic_enemy_labels(labels, counts, objects, bars, min_hits=1)
        self.assertIn("goblin", updated)


if __name__ == "__main__":
    unittest.main()
