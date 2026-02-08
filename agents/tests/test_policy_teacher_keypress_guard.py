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

    def test_noncombat_keypress_prefers_meaningful_fallback(self):
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
        agent._meaningful_fallback_action = lambda _state: {
            "label": "mouse_move",
            "dx": 32,
            "dy": -18,
            "target_norm": [0.62, 0.44],
        }

        chosen = agent._blend_with_teacher({"label": "key_press", "key": "2"})

        self.assertEqual(chosen.get("label"), "mouse_move")
        self.assertEqual(chosen.get("reason"), "suppress_noncombat_keypress")

    def test_generic_teacher_wait_does_not_override_non_wait_policy_action(self):
        agent = PolicyAgent()
        agent.stage0_enabled = False
        agent.current_intent = "observe"
        agent.latest_state = {"flags": {"in_game": True, "death": False}}
        agent._teacher_to_action = lambda: {"label": "wait"}

        chosen = agent._blend_with_teacher({"label": "click_primary", "reason": "option_explore"})

        self.assertEqual(chosen.get("label"), "click_primary")

    def test_policy_wait_in_game_uses_option_layer_before_generic_teacher(self):
        agent = PolicyAgent()
        agent.stage0_enabled = False
        agent.current_intent = "observe"
        agent.latest_state = {"flags": {"in_game": True, "death": False}}
        agent._teacher_to_action = lambda: {"label": "wait"}
        agent._option_layer_action = lambda _state: {
            "label": "mouse_move",
            "source": "option_layer",
            "reason": "option_explore",
            "dx": 5,
            "dy": -3,
        }

        chosen = agent._blend_with_teacher({"label": "wait"})

        self.assertEqual(chosen.get("label"), "mouse_move")
        self.assertEqual(chosen.get("source"), "option_layer")


if __name__ == "__main__":
    unittest.main()
