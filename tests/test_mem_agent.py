import shutil
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

import agents.mem_agent as mem


class MemAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_rules = mem.rules
        self._orig_rules_path = mem.rules_path
        self._orig_recent = mem.recent_critical
        self._orig_decay = mem.RULE_DECAY_SEC
        self._orig_penalty = mem.RULE_DECAY_PENALTY
        self._orig_retire = mem.RULE_RETIRE_THRESHOLD
        mem.rules = []
        mem.rules_path = self.tmpdir / "rules.json"
        mem.recent_critical = deque(maxlen=2)
        mem.RULE_DECAY_SEC = 0.01
        mem.RULE_DECAY_PENALTY = 0.6
        mem.RULE_RETIRE_THRESHOLD = 0.2

    def tearDown(self):
        mem.rules = self._orig_rules
        mem.rules_path = self._orig_rules_path
        mem.recent_critical = self._orig_recent
        mem.RULE_DECAY_SEC = self._orig_decay
        mem.RULE_DECAY_PENALTY = self._orig_penalty
        mem.RULE_RETIRE_THRESHOLD = self._orig_retire
        shutil.rmtree(self.tmpdir)

    def test_recent_critical_fifo(self):
        mem.insert_recent_critical({"episode_id": "a", "delta": "hero_dead"})
        mem.insert_recent_critical({"episode_id": "b", "delta": "hero_resurrected"})
        mem.insert_recent_critical({"episode_id": "c", "delta": "ui_change"})
        result = mem.query({"mode": "recent_critical", "limit": 5})
        ids = [entry["episode_id"] for entry in result["value"]]
        self.assertEqual(ids, ["c", "b"], ids)

    def test_rule_usage_and_decay(self):
        mem.insert_rule({"rule_id": "r1", "scope": "death_dialog", "text": "stay calm", "confidence": 0.9})
        mem.mark_rule_used("r1", timestamp=time.time())
        self.assertEqual(mem.rules[0]["usage_count"], 1)
        self.assertFalse(mem.rules[0].get("retired"))
        mem.rules[0]["last_used_at"] = time.time() - 1.0
        mem.rules[0]["last_decay"] = mem.rules[0]["last_used_at"]
        mem.apply_rule_decay(now=time.time())
        mem.rules[0]["last_decay"] = mem.rules[0]["last_used_at"]
        mem.apply_rule_decay(now=time.time())
        self.assertTrue(mem.rules[0]["retired"], mem.rules[0])

    def test_calibration_events(self):
        mem.insert_calibration_event(
            {"profile": "p1", "scope": "critical_dialog:death", "x_norm": 0.5, "y_norm": 0.8}
        )
        mem.insert_calibration_event(
            {"profile": "other", "scope": "generic", "x_norm": 0.1, "y_norm": 0.1}
        )
        response = mem.query(
            {
                "mode": "calibration_events",
                "scope": "critical_dialog:death",
                "profile": "p1",
                "limit": 5,
            }
        )
        self.assertEqual(len(response["value"]), 1)
        self.assertEqual(response["value"][0]["profile"], "p1")


if __name__ == "__main__":
    unittest.main()
