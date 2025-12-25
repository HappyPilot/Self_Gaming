import unittest

from env_api.in_memory import InMemoryEnvAdapter


class EnvAdapterTest(unittest.TestCase):
    def test_step_returns_observation(self):
        env = InMemoryEnvAdapter()
        env.push_observation({"ok": True, "frame_id": 1, "timestamp": 123.4})
        result = env.step({"action": "mouse_move", "dx": 1, "dy": 2})
        self.assertIsNotNone(result.observation)
        self.assertEqual(result.observation.frame_id, 1)


if __name__ == "__main__":
    unittest.main()
