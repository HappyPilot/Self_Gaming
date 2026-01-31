import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "agents"))

from utils.combat_targeting import pick_enemy_target


class CombatTargetingTest(unittest.TestCase):
    def test_cluster_targeting(self):
        enemies = [
            {"bbox": [0.4, 0.4, 0.45, 0.45]},
            {"bbox": [0.5, 0.4, 0.55, 0.45]},
        ]
        target = pick_enemy_target(enemies, player_center=(0.5, 0.5))
        self.assertIsNotNone(target)


if __name__ == "__main__":
    unittest.main()
