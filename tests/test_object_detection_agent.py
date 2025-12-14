import base64
import unittest

import cv2
import numpy as np

from agents.object_detection_agent import Detection, DummyDetector, decode_frame


class ObjectDetectionAgentTest(unittest.TestCase):
    def test_decode_frame_roundtrip(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        ok, jpg = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

        decoded = decode_frame(b64)
        self.assertEqual(decoded.shape[0], 4)
        self.assertEqual(decoded.shape[1], 4)

    def test_detection_dataclass(self):
        det = Detection(label="monster", confidence=0.98765, box=[1.2345, 2.3456, 3.4567, 4.5678])
        payload = det.as_dict()
        self.assertEqual(payload["class"], "monster")
        self.assertAlmostEqual(payload["confidence"], round(0.98765, 4), places=4)
        self.assertEqual(len(payload["box"]), 4)

    def test_dummy_detector_returns_empty(self):
        detector = DummyDetector()
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        self.assertEqual(list(detector.detect(frame)), [])


if __name__ == "__main__":
    unittest.main()
