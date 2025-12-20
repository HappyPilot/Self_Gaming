import unittest

from agents.utils.latency import build_control_event, build_latency_event


class LatencyUtilsTest(unittest.TestCase):
    def test_build_latency_event_with_sla(self):
        event = build_latency_event(
            "detect",
            12.3456,
            sla_ms=20,
            tags={"agent": "object_detection_agent"},
            agent="object_detection_agent",
            timestamp=1.23,
        )
        self.assertEqual(event["event"], "latency")
        self.assertEqual(event["stage"], "detect")
        self.assertEqual(event["timestamp"], 1.23)
        self.assertTrue(event["ok"])
        self.assertIn("sla_ms", event)
        self.assertIn("tags", event)

    def test_build_latency_event_without_sla(self):
        event = build_latency_event("fuse", 50.0)
        self.assertEqual(event["event"], "latency")
        self.assertEqual(event["stage"], "fuse")
        self.assertNotIn("ok", event)

    def test_build_control_event(self):
        event = build_control_event(
            "control/tick_ms",
            12.3,
            ok=True,
            tags={"window": 50},
            timestamp=2.34,
        )
        self.assertEqual(event["event"], "control_metric")
        self.assertEqual(event["metric"], "control/tick_ms")
        self.assertEqual(event["value"], 12.3)
        self.assertEqual(event["timestamp"], 2.34)
        self.assertTrue(event["ok"])
        self.assertIn("tags", event)


if __name__ == "__main__":
    unittest.main()
