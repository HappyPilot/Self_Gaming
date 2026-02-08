import unittest

from tools.mac import input_helpers as helpers


class LaptopInputTargetTest(unittest.TestCase):
    def test_resolve_target_norm_without_bounds(self):
        point = helpers.resolve_target_point({"target_norm": [0.5, 0.25]}, None, (200, 100))
        self.assertEqual(point, (100, 25))

    def test_resolve_target_norm_with_bounds(self):
        bounds = (10, 20, 110, 220)
        point = helpers.resolve_target_point({"target_norm": [0.5, 0.5]}, bounds, (200, 100))
        self.assertEqual(point, (60, 120))

    def test_resolve_target_px_clamped(self):
        bounds = (0, 0, 100, 100)
        point = helpers.resolve_target_point({"target_px": [200, -5]}, bounds, (200, 100))
        self.assertEqual(point, (100, 0))


if __name__ == "__main__":
    unittest.main()
