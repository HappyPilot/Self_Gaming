Replay Harness
==============

Replay a recorded session back into MQTT, using the same topics that were
captured by the recorder session.

Quick start
-----------

```bash
python -m replay.replay_runner /mnt/ssd/datasets/<game>/<session_id>
```

Common options
--------------

```bash
REPLAY_SPEED=2.0 REPLAY_MAX_SEC=20 \
python -m replay.replay_runner /mnt/ssd/datasets/<game>/<session_id>
```

- `REPLAY_SPEED` - playback speed multiplier (default 1.0).
- `REPLAY_MAX_SEC` - cap playback duration in seconds (0 = full session).
- `REPLAY_START_DELAY_SEC` - delay before publishing (default 0.5).
- `REPLAY_FRAME_TOPIC` - override the frame topic.
- `REPLAY_SENSOR_TOPICS` - comma-separated sensor topics to include.
- `REPLAY_PUBLISH_FRAMES` / `REPLAY_PUBLISH_SENSORS` - toggle streams.

Notes
-----

- For clean replays, disable live sensor agents and consume the replayed
  `sensors.jsonl` stream instead.
