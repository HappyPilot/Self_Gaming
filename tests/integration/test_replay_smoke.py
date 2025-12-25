import json
import tempfile
import unittest
from pathlib import Path

from replay.replay_runner import ReplayRunner


class TestReplaySmoke(unittest.TestCase):
    def test_replay_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"
            frames_dir = session_dir / "frames"
            frames_dir.mkdir(parents=True)

            meta = {"frame_topic": "vision/frame/preview"}
            (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            frame_path = frames_dir / "1700000000000_000001.jpg"
            frame_path.write_bytes(b"\xff\xd8\xff\xd9")

            sensor = {
                "timestamp": 1700000000.1,
                "topic": "scene/state",
                "payload": {"ok": True},
            }
            (session_dir / "sensors.jsonl").write_text(
                json.dumps(sensor) + "\n",
                encoding="utf-8",
            )

            runner = ReplayRunner(
                session_dir,
                speed=1000.0,
                max_sec=1.0,
                start_delay=0.0,
                dry_run=True,
            )
            stats = runner.run()

            self.assertEqual(stats.frames_published, 1)
            self.assertEqual(stats.sensors_published, 1)


if __name__ == "__main__":
    unittest.main()
