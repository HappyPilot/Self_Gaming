import unittest

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


class PolicyTaskMoveCompletionTest(unittest.TestCase):
    def setUp(self):
        self._old_click_to_move = policy.POLICY_CLICK_TO_MOVE
        policy.POLICY_CLICK_TO_MOVE = True

    def tearDown(self):
        policy.POLICY_CLICK_TO_MOVE = self._old_click_to_move

    def test_move_to_task_completes_on_click_when_click_to_move(self):
        agent = PolicyAgent()
        task = {
            "goal_id": "respawn_123",
            "task_id": "respawn_move_123",
            "action_type": "MOVE_TO",
            "target": {"x_norm": 0.5, "y_norm": 0.82},
            "status": "pending",
        }
        agent.task_queue.clear()
        agent.task_queue.append(task)
        agent.current_task = task
        agent.respawn_pending = True

        action = agent._action_from_task({})
        self.assertEqual(action.get("label"), "click_primary")

        agent._maybe_advance_task(action)
        self.assertIsNone(agent.current_task)
        self.assertFalse(agent.respawn_pending)


if __name__ == "__main__":
    unittest.main()
