import unittest
from agents.utils.control_profile_v2 import normalize_profile_v2


class ControlProfileV2Test(unittest.TestCase):
    def test_normalize_profile_groups(self):
        profile = {
            "game_id": "test_game",
            "bindings": [
                {"action": "inventory_open", "keys": ["i"], "category": "ui"},
                {"action": "skill_1", "keys": ["q"], "category": "combat"},
                {"action": "special_combo", "keys": ["shift+1"], "category": "combat"},
            ],
        }
        normalized = normalize_profile_v2(profile)
        self.assertIn("allowed_keys_gameplay", normalized)
        self.assertIn("blocked_keys_aux", normalized)
        self.assertIn("allowed_combos_gameplay", normalized)
        self.assertIn("i", normalized["blocked_keys_aux"])
        self.assertIn("q", normalized["allowed_keys_gameplay"])
        self.assertIn("shift+1", normalized["allowed_combos_gameplay"])


if __name__ == "__main__":
    unittest.main()
