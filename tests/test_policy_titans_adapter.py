import importlib.util
import os
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TITANS = importlib.util.find_spec("titans_pytorch") is not None


@unittest.skipUnless(HAS_TORCH and HAS_TITANS, "titans_pytorch/torch not available")
class TitansPolicyAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {}
        for key in ("TITANS_DEVICE", "TITANS_UPDATE_INTERVAL"):
            self._env[key] = os.getenv(key)
        os.environ["TITANS_DEVICE"] = "cpu"
        os.environ["TITANS_UPDATE_INTERVAL"] = "1"

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_predict_chunk_basic(self) -> None:
        from sg_platform.policy_titans_adapter import TitansPolicyAdapter

        adapter = TitansPolicyAdapter(action_space_dim=3, device="cpu", dim=8, chunk_size=2)
        chunk = adapter.predict_chunk({}, {"latent_state": [0.0] * 8})

        self._assert_valid_chunk(chunk, expected_dim=3)

    def test_predict_chunk_repeatable(self) -> None:
        from sg_platform.policy_titans_adapter import TitansPolicyAdapter

        adapter = TitansPolicyAdapter(action_space_dim=2, device="cpu", dim=8, chunk_size=2)
        first = adapter.predict_chunk({}, {"latent_state": [0.1] * 8})
        second = adapter.predict_chunk({}, {"latent_state": [0.2] * 8})
        self._assert_valid_chunk(first, expected_dim=2)
        self._assert_valid_chunk(second, expected_dim=2)

    def test_update_interval_buffers_latents(self) -> None:
        from sg_platform.policy_titans_adapter import TitansPolicyAdapter

        os.environ["TITANS_UPDATE_INTERVAL"] = "2"
        adapter = TitansPolicyAdapter(action_space_dim=2, device="cpu", dim=8, chunk_size=2)
        first = adapter.predict_chunk({}, {"latent_state": [0.3] * 8})
        second = adapter.predict_chunk({}, {"latent_state": [0.4] * 8})
        self._assert_valid_chunk(first, expected_dim=2)
        self._assert_valid_chunk(second, expected_dim=2)

    def _assert_valid_chunk(self, chunk: dict, expected_dim: int) -> None:
        self.assertIsInstance(chunk, dict)
        self.assertEqual(chunk.get("horizon"), 1)
        actions = chunk.get("actions")
        self.assertIsInstance(actions, list)
        self.assertEqual(len(actions), 1)
        vector = actions[0].get("vector")
        self.assertEqual(len(vector), expected_dim)


if __name__ == "__main__":
    unittest.main()
