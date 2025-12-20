import base64
import unittest

from agents.utils.frame_transport import SHM_AVAILABLE, ShmFrameRing, get_frame_b64, get_frame_bytes


class FrameTransportTest(unittest.TestCase):
    def test_base64_roundtrip(self):
        payload = {"image_b64": base64.b64encode(b"hello").decode("ascii")}
        self.assertEqual(get_frame_bytes(payload), b"hello")
        self.assertEqual(get_frame_b64(payload), payload["image_b64"])

    @unittest.skipUnless(SHM_AVAILABLE, "shared_memory unavailable")
    def test_shm_roundtrip(self):
        ring = ShmFrameRing(max_bytes=1024, slots=1, prefix="sg_test")
        try:
            data = b"frame_bytes"
            desc = ring.write(data)
            self.assertIsNotNone(desc)
            payload = {"transport": "shm", **desc}
            self.assertEqual(get_frame_bytes(payload), data)
        finally:
            ring.close()


if __name__ == "__main__":
    unittest.main()
