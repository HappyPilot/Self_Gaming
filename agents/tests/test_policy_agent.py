import unittest

from agents.policy_agent import PolicyAgent, compute_cursor_motion


class PolicyAgentTest(unittest.TestCase):
    def test_teacher_alpha_anneals_linearly(self):
        agent = PolicyAgent()
        agent.teacher_alpha_start = 1.0
        agent.teacher_decay_steps = 10
        agent.teacher_min_alpha = 0.0

        agent.steps = 0
        self.assertAlmostEqual(agent._current_alpha(), 1.0)
        agent.steps = 5
        self.assertAlmostEqual(agent._current_alpha(), 0.5)
        agent.steps = 10
        self.assertAlmostEqual(agent._current_alpha(), 0.0)
        agent.steps = 20
        self.assertAlmostEqual(agent._current_alpha(), 0.0)

    def test_compute_cursor_motion_includes_offsets(self):
        target_px, cursor_px, delta = compute_cursor_motion(
            x_norm=0.75,
            y_norm=0.25,
            cursor_x_norm=0.5,
            cursor_y_norm=0.5,
            width=1000,
            height=800,
            offset_x=10,
            offset_y=-20,
        )
        self.assertEqual(target_px, (760, 180))
        self.assertEqual(cursor_px, (510, 380))
        self.assertEqual(delta, (250, -200))


if __name__ == "__main__":
    unittest.main()
