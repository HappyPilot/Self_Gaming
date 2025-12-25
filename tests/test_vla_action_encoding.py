import unittest

from training.vla.utils import encode_action


class VLAActionEncodingTest(unittest.TestCase):
    def test_action_dim_truncates(self):
        payload = {"dx": 1.0, "dy": -2.0, "button": 1, "key": "space"}
        encoded = encode_action(payload, action_dim=2)
        self.assertEqual(encoded, [1.0, -2.0])

    def test_action_dim_pads(self):
        payload = {"dx": 0.5, "dy": 0.25}
        encoded = encode_action(payload, action_dim=6)
        self.assertEqual(encoded, [0.5, 0.25, 0.0, 0.0, 0.0, 0.0])

    def test_button_zero_counts_as_click(self):
        payload = {"dx": 0.0, "dy": 0.0, "button": 0}
        encoded = encode_action(payload, action_dim=4)
        self.assertEqual(encoded[2], 1.0)


if __name__ == "__main__":
    unittest.main()
