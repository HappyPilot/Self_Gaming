import unittest

from agents.scene_agent import SceneAgent


class SceneAgentEnemyExtractionTest(unittest.TestCase):
    def test_extract_enemies_uses_class_when_label_missing(self):
        agent = SceneAgent()
        objects = [
            {
                "class": "monster",
                "confidence": 0.91,
                "bbox": [0.1, 0.1, 0.2, 0.2],
            }
        ]

        enemies = agent._extract_enemies(objects)

        self.assertEqual(len(enemies), 1)
        self.assertEqual(enemies[0].get("label"), "monster")


if __name__ == "__main__":
    unittest.main()
