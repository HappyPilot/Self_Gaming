import unittest

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


class PolicyTaskClickTargetTest(unittest.TestCase):
    def test_attack_target_includes_target_norm(self):
        agent = PolicyAgent()
        agent.current_task = {
            "action_type": "ATTACK_TARGET",
            "task_id": "task_test",
            "target": {"x_norm": 0.25, "y_norm": 0.75},
        }
        action = agent._action_from_task({})
        self.assertEqual(action.get("label"), "click_primary")
        self.assertEqual(action.get("target_norm"), [0.25, 0.75])

    def test_move_to_click_to_move(self):
        policy.POLICY_CLICK_TO_MOVE = True
        agent = PolicyAgent()
        agent.current_task = {
            "action_type": "MOVE_TO",
            "task_id": "task_move",
            "target": {"x_norm": 0.4, "y_norm": 0.6},
        }
        action = agent._action_from_task({})
        self.assertEqual(action.get("label"), "click_primary")
        self.assertEqual(action.get("target_norm"), [0.4, 0.6])
        policy.POLICY_CLICK_TO_MOVE = False

    def test_exploration_click_includes_target_norm(self):
        old_allow = policy.POLICY_EXPLORATION_ALLOW_CLICK
        old_burst = policy.POLICY_EXPLORATION_BURST_ACTIONS
        policy.POLICY_EXPLORATION_ALLOW_CLICK = True
        policy.POLICY_EXPLORATION_BURST_ACTIONS = 1
        agent = PolicyAgent()
        agent._exploration_targets = lambda: [(0.2, 0.3)]
        agent._queue_exploration_burst()
        actions = list(agent.exploration_queue)
        self.assertTrue(actions, "expected exploration actions")
        click = actions[-1]
        self.assertEqual(click.get("label"), "click_primary")
        self.assertEqual(click.get("target_norm"), [0.2, 0.3])
        policy.POLICY_EXPLORATION_ALLOW_CLICK = old_allow
        policy.POLICY_EXPLORATION_BURST_ACTIONS = old_burst


if __name__ == "__main__":
    unittest.main()
