import unittest

from agents.motion_anchor_agent import build_anchor_payload, resolve_backend


class MotionAnchorAgentTest(unittest.TestCase):
    def test_build_anchor_payload(self):
        payload = build_anchor_payload(0.25, 0.75, 0.5, ts=123.0)
        self.assertEqual(payload["label"], "motion_salient")
        self.assertEqual(payload["point"], [0.25, 0.75])
        self.assertEqual(payload["score"], 0.5)
        self.assertEqual(payload["ts"], 123.0)

    def test_resolve_backend_prefers_vpi(self):
        backend = resolve_backend("vpi", vpi_available=True)
        self.assertEqual(backend, "vpi")
        backend = resolve_backend("vpi", vpi_available=False)
        self.assertEqual(backend, "cpu")


if __name__ == "__main__":
    unittest.main()
