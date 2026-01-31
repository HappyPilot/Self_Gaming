import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "agents"))

from agents.skill_profiler_agent import classify_skill_type


class SkillProfilerTest(unittest.TestCase):
    def test_classify_skill_type(self):
        result = classify_skill_type("Deals damage in an area around you")
        self.assertEqual(result, "aoe")


if __name__ == "__main__":
    unittest.main()
