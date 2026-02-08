import unittest

import agents.zeroshot_detector as zsd


class _DummyProcessor:
    def __init__(self):
        self.called_threshold = None

    def post_process_object_detection(self, outputs=None, target_sizes=None, threshold=None):
        self.called_threshold = threshold
        return [{"scores": []}]


class ZeroshotDetectorTest(unittest.TestCase):
    def test_post_process_uses_configured_threshold(self):
        processor = _DummyProcessor()
        old_threshold = zsd.POST_THRESHOLD
        try:
            zsd.POST_THRESHOLD = 0.01
            zsd._post_process(processor, outputs=None, target_sizes=None)
            self.assertEqual(processor.called_threshold, 0.01)
        finally:
            zsd.POST_THRESHOLD = old_threshold

    def test_format_object_adds_bbox_and_center(self):
        obj = zsd._format_object("enemy", 0.5, [10.0, 20.0, 110.0, 220.0], (200, 400))
        self.assertEqual(obj["bbox"], [0.05, 0.05, 0.55, 0.55])
        self.assertEqual(obj["center"], [0.3, 0.3])


if __name__ == "__main__":
    unittest.main()
