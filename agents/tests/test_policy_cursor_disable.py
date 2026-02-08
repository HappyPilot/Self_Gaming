import importlib
import os
import sys
import time
import unittest


def _reload_policy():
    if "agents.policy_agent" in sys.modules:
        del sys.modules["agents.policy_agent"]
    return importlib.import_module("agents.policy_agent")


class PolicyCursorDisableTest(unittest.TestCase):
    def test_cursor_disabled_returns_not_fresh(self):
        os.environ["POLICY_USE_CURSOR"] = "0"
        mod = _reload_policy()
        agent = mod.PolicyAgent()
        agent.cursor_detected_ts = time.time()
        self.assertFalse(agent._cursor_is_fresh())


if __name__ == "__main__":
    unittest.main()
