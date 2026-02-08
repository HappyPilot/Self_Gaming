import importlib
import os
import sys
import unittest

import numpy as np


def _reload_cursor_tracker():
    if "agents.cursor_tracker_agent" in sys.modules:
        del sys.modules["agents.cursor_tracker_agent"]
    return importlib.import_module("agents.cursor_tracker_agent")


class CursorTrackerMaskTest(unittest.TestCase):
    def test_masked_region_ignored(self):
        os.environ["CURSOR_MASK_TOP"] = "0.3"
        os.environ["CURSOR_MASK_RIGHT"] = "0.3"
        mod = _reload_cursor_tracker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0:10, 85:95] = 255  # bright blob in top-right
        detection, _ = mod.detect_cursor(frame, last_point=None, prev_gray=None)
        self.assertIsNone(detection)


if __name__ == "__main__":
    unittest.main()
