import unittest

from agents.policy_agent import PolicyAgent


class PolicyTeacherKeypressGuardTest(unittest.TestCase):
    def test_stage0_does_not_force_teacher_keypress_outside_combat(self):
        agent = PolicyAgent()
        agent.stage0_enabled = True
        agent.current_intent = "observe"
        agent.latest_state = {
            "flags": {"in_game": True, "death": False},
            "enemies": [],
            "objects": [],
            "targets": [],
            "stats": {"enemy_count": 0},
        }
        agent._teacher_to_action = lambda: {"label": "key_press", "key": "1", "text": "press 1"}
        agent._enemies_present = lambda: False
        agent._latest_scene_has_respawn_text = lambda: False

        chosen = agent._blend_with_teacher({"label": "click_primary", "target_norm": [0.5, 0.5]})

        self.assertEqual(chosen.get("label"), "click_primary")

    def test_policy_keypress_is_suppressed_outside_combat(self):
        agent = PolicyAgent()
        agent.stage0_enabled = False
        agent.current_intent = "observe"
        agent.latest_state = {
            "flags": {"in_game": True, "death": False},
            "enemies": [],
            "objects": [],
            "targets": [],
            "stats": {"enemy_count": 0},
        }
        agent._teacher_to_action = lambda: None
        agent._enemies_present = lambda: False
        agent._latest_scene_has_respawn_text = lambda: False

        chosen = agent._blend_with_teacher({"label": "key_press", "key": "1"})

        self.assertIn(chosen.get("label"), {"click_primary", "mouse_move"})


if __name__ == "__main__":
    unittest.main()
