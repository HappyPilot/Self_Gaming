import unittest

from agents.reward_manager import RewardCalculator, SceneMetrics, classify_item


class RewardManagerTest(unittest.TestCase):
    def test_classify_item_currency(self):
        meta = classify_item("Exalted Orb")
        self.assertEqual(meta["bucket"], "currency_high")
        self.assertGreater(meta["value"], 0)

    def test_reward_positive_for_progress(self):
        calc = RewardCalculator(stage="S1")
        events = [
            {"type": "area", "meta": {"area": "The Coast"}},
            {"type": "kill", "meta": {"class": "elite"}},
            {"type": "loot", "meta": {"value": 5.0}},
        ]
        reward, components = calc.compute(events, SceneMetrics(enemy_count=0, loot_score=0.0))
        self.assertGreater(reward, 0)
        self.assertIn("total", components)

    def test_death_penalty_negative(self):
        calc = RewardCalculator(stage="S3")
        events = [{"type": "death", "meta": {}}]
        reward, _ = calc.compute(events, SceneMetrics())
        self.assertLess(reward, 0)


if __name__ == "__main__":
    unittest.main()
