import unittest

from agents.policy_agent import PolicyAgent


class PolicyTeacherKeyConfirmTest(unittest.TestCase):
    def _make_agent(self):
        agent = PolicyAgent()
        agent.teacher_alpha_start = 1.0
        agent.teacher_decay_steps = 1000
        agent.teacher_min_alpha = 0.0
        agent.steps = 0
        agent.stage0_enabled = False
        agent.latest_state = {"flags": {}}
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
