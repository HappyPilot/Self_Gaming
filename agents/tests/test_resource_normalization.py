import math
import unittest

from agents.policy_agent import encode_non_visual
from agents.scene_agent import SceneAgent
from agents.train_manager_agent import TrainManagerAgent


class ResourceNormalizationTest(unittest.TestCase):
    def test_scene_agent_extracts_plain_value_with_name(self):
        agent = SceneAgent()
        resources = agent._extract_resources({"hud": {"text": "Score: 123"}})
        self.assertIn("score", resources)
        info = resources["score"]
        self.assertEqual(info.get("current"), 123)
        self.assertEqual(info.get("max"), 123)
        self.assertTrue(info.get("is_plain_value"))

    def test_scene_agent_ignores_years_and_months(self):
        agent = SceneAgent()
        resources = agent._extract_resources({"hud": {"text": "February 2026"}})
        self.assertFalse(resources, "Months/years should be ignored in resources")

        resources = agent._extract_resources({"hud": {"text": "2020"}})
        self.assertFalse(resources, "Standalone years should be ignored in resources")

    def test_policy_normalizes_plain_value_percent_log(self):
        state = {
            "resources": {
                "score": {"current": 80, "max": 80, "is_plain_value": True},
            }
        }
        vector = encode_non_visual(state)
        self.assertAlmostEqual(vector[5].item(), 0.8, places=5)

        state = {
            "resources": {
                "score": {"current": 123, "max": 123, "is_plain_value": True},
            }
        }
        vector = encode_non_visual(state)
        expected = math.log1p(123) / math.log1p(1000)
        self.assertAlmostEqual(vector[5].item(), expected, places=5)

    def test_train_manager_normalizes_plain_value_percent_log(self):
        agent = TrainManagerAgent()
        scene = {
            "resources": {
                "score": {"current": 150, "max": 150, "is_plain_value": True},
            }
        }
        vector = agent._encode_scene(scene)
        expected = math.log1p(150) / math.log1p(1000)
        self.assertAlmostEqual(float(vector[5]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
