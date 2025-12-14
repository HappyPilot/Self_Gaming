import unittest

from agents.policy_agent import PolicyAgent


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


if __name__ == "__main__":
    unittest.main()
