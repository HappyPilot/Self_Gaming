import unittest

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


class PolicyEnemyPresenceConfidenceTest(unittest.TestCase):
    def setUp(self):
        self._old_min_conf = getattr(policy, "POLICY_ENEMY_MIN_CONF", None)
        policy.POLICY_ENEMY_MIN_CONF = 0.2

    def tearDown(self):
        if self._old_min_conf is None:
            delattr(policy, "POLICY_ENEMY_MIN_CONF")
        else:
            policy.POLICY_ENEMY_MIN_CONF = self._old_min_conf

    def test_enemies_present_respects_confidence_threshold(self):
        agent = PolicyAgent()
        agent.latest_state = {
            "enemies": [
                {"label": "monster", "confidence": 0.05, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        }
        self.assertFalse(agent._enemies_present())

        agent.latest_state = {
            "enemies": [
                {"label": "monster", "confidence": 0.35, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        }
        self.assertTrue(agent._enemies_present())

    def test_enemies_present_in_state_respects_confidence_threshold(self):
        state = {
            "enemies": [
                {"label": "monster", "confidence": 0.05, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        }
        self.assertFalse(policy.PolicyAgent._enemies_present_in_state(state))

        state = {
            "enemies": [
                {"label": "monster", "confidence": 0.35, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        }
        self.assertTrue(policy.PolicyAgent._enemies_present_in_state(state))


if __name__ == "__main__":
    unittest.main()
