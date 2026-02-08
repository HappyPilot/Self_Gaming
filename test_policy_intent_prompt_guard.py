import unittest

from agents.policy_agent import PolicyAgent


class PolicyIntentPromptGuardTest(unittest.TestCase):
    def test_enemy_prompt_requires_enemy_evidence(self):
        agent = PolicyAgent()
        agent.intent_enemy_score = 0.01
        state = {
            "flags": {"in_game": True},
            "prompt_scores": {"boss": 0.03, "enemy": 0.02},
            "text": ["ENEMY AT THE GATE", "Kill Hillock"],
            "objects": [],
            "targets": [],
            "enemies": [],
            "stats": {"enemy_count": 0},
        }

        intent, reason = agent._infer_intent(state)

        self.assertEqual(intent, "observe")
        self.assertEqual(reason, "default")


if __name__ == "__main__":
    unittest.main()
