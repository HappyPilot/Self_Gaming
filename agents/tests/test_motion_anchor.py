import unittest
import numpy as np

from agents.utils.motion_anchor import compute_motion_center


class MotionAnchorTest(unittest.TestCase):
    def test_detects_motion_center(self):
        h, w = 40, 60
        prev = np.zeros((h, w), dtype=np.uint8)
        curr = np.zeros((h, w), dtype=np.uint8)
        prev[10:14, 10:14] = 255
        curr[10:14, 16:20] = 255  # shift right
        center = compute_motion_center(prev, curr, mag_threshold=5, min_mean=0.5)
        self.assertIsNotNone(center)
        cx, cy, score = center
        self.assertGreater(score, 0.0)
        self.assertAlmostEqual(cx, 0.3, delta=0.1)
        self.assertAlmostEqual(cy, 0.3, delta=0.1)

    def test_ignores_static_frames(self):
        h, w = 40, 60
        frame = np.zeros((h, w), dtype=np.uint8)
        center = compute_motion_center(frame, frame, mag_threshold=5, min_mean=0.5)
        self.assertIsNone(center)


if __name__ == "__main__":
    unittest.main()
