import unittest

from reflex_policy import ReflexPolicyAdapter


class TestReflexPolicyAdapter(unittest.TestCase):
    def test_predict_returns_none(self):
        adapter = ReflexPolicyAdapter()
        observation = {"ok": True, "timestamp": 123.0}
        strategy_state = {"global_strategy": {}, "targets": {}, "cooldowns": {}}
        self.assertIsNone(adapter.predict(observation, strategy_state))


if __name__ == "__main__":
    unittest.main()
