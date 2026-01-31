import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "agents"))

from agents.game_onboarding_agent import GameOnboardingAgent


class OnboardingProfileV2Test(unittest.TestCase):
    def test_profile_to_controls_uses_gameplay_keys_only(self):
        agent = GameOnboardingAgent()
        profile = {
            "allow_mouse_move": True,
            "allow_primary": True,
            "bindings": [
                {"action": "inventory_open", "keys": ["i"], "category": "ui"},
                {"action": "skill_1", "keys": ["q"], "category": "combat"},
            ],
        }
        controls = agent._profile_to_controls(profile)
        self.assertIn("key_q", controls)
        self.assertNotIn("key_i", controls)


if __name__ == "__main__":
    unittest.main()
