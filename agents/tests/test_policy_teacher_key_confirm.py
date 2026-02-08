import unittest

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


class PolicyTeacherKeyConfirmTest(unittest.TestCase):
    def setUp(self):
        self._old_enemy_skill_sub = policy.POLICY_ENEMY_SKILL_SUBSTITUTE
        policy.POLICY_ENEMY_SKILL_SUBSTITUTE = False

    def tearDown(self):
        policy.POLICY_ENEMY_SKILL_SUBSTITUTE = self._old_enemy_skill_sub

    def _make_agent(self):
        agent = PolicyAgent()
        agent.teacher_alpha_start = 1.0
        agent.teacher_decay_steps = 1000
        agent.teacher_min_alpha = 0.0
        agent.steps = 0
        agent.stage0_enabled = False
        agent.current_intent = "combat"
        agent.latest_state = {"flags": {"in_game": True, "death": False}, "stats": {"enemy_count": 1}}
        return agent

    def test_teacher_key_press_blocked_until_confirmed(self):
        agent = self._make_agent()
        agent.profile_allowed_keys = set()
        agent.skill_stats = {}
        agent.teacher_action = {
            "text": "press q",
            "action": {"label": "key_press", "key": "q"},
            "timestamp": 0.0,
        }

        chosen = agent._blend_with_teacher({"label": "wait"})

        self.assertNotEqual(chosen.get("label"), "key_press")

    def test_teacher_key_press_allowed_after_confirmed(self):
        agent = self._make_agent()
        agent.profile_allowed_keys = set()
        agent.skill_stats = {"q": {"hits": 1.0, "tries": 1.0}}
        agent.teacher_action = {
            "text": "press q",
            "action": {"label": "key_press", "key": "q"},
            "timestamp": 0.0,
        }

        chosen = agent._blend_with_teacher({"label": "wait"})

        self.assertEqual(chosen.get("label"), "key_press")
        self.assertEqual(chosen.get("key"), "q")


if __name__ == "__main__":
    unittest.main()
