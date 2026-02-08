import unittest
import time

from agents.policy_agent import PolicyAgent


class PolicyMotionAnchorTest(unittest.TestCase):
    def test_motion_anchor_action_moves(self):
        agent = PolicyAgent()
        agent.motion_anchor = {"point": [0.7, 0.3], "score": 0.5, "ts": time.time()}
        action = agent._motion_anchor_action()
        self.assertIsNotNone(action)
        self.assertEqual(action["label"], "mouse_move")

    def test_motion_anchor_stale_ignored(self):
        agent = PolicyAgent()
        agent.motion_anchor = {"point": [0.7, 0.3], "score": 0.5, "ts": time.time() - 10}
        action = agent._motion_anchor_action()
        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main()
