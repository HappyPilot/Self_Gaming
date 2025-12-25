import unittest

from control.rtc_executor import RTCExecutor


class RTCExecutorTest(unittest.TestCase):
    def test_gap_fill_when_missing_next_chunk(self):
        actions = []

        def publish(action):
            actions.append(action)

        executor = RTCExecutor(publish_action=publish, overlap_steps=0, gap_fill_mode="zero_move")
        executor.enqueue_chunk([{"action": "mouse_move", "dx": 1, "dy": 1}])

        first = executor.tick()
        gap = executor.tick()

        self.assertEqual(first.get("action"), "mouse_move")
        self.assertEqual(gap.get("action"), "mouse_move")
        self.assertEqual(gap.get("dx"), 0)
        self.assertEqual(gap.get("dy"), 0)

    def test_overlap_blend_actions(self):
        actions = []

        def publish(action):
            actions.append(action)

        executor = RTCExecutor(publish_action=publish, overlap_steps=2, gap_fill_mode="zero_move")
        executor.enqueue_chunk([{"action": "mouse_move", "dx": 0, "dy": 0}])
        executor.enqueue_chunk([{"action": "mouse_move", "dx": 10, "dy": 0}])

        executor.tick()  # current chunk
        executor.tick()  # blend 1
        executor.tick()  # blend 2
        executor.tick()  # first next action

        self.assertEqual(len(actions), 4)
        self.assertEqual(actions[0]["dx"], 0)
        self.assertAlmostEqual(actions[1]["dx"], 10 / 3, places=2)
        self.assertAlmostEqual(actions[2]["dx"], 20 / 3, places=2)
        self.assertEqual(actions[3]["dx"], 10)

    def test_ready_ratio_metric_emitted(self):
        metrics = []

        def emit_metric(name, value, ok, tags):
            metrics.append((name, value, tags))

        def publish(_action):
            return None

        executor = RTCExecutor(
            publish_action=publish,
            overlap_steps=0,
            sample_every=1,
            ready_window=3,
            metric_emitter=emit_metric,
        )
        executor.enqueue_chunk([{"action": "mouse_move", "dx": 0, "dy": 0}])
        executor.enqueue_chunk([{"action": "mouse_move", "dx": 1, "dy": 1}])

        executor.tick()

        names = [item[0] for item in metrics]
        self.assertIn("control/next_chunk_ready_ratio", names)


if __name__ == "__main__":
    unittest.main()
