Recording Format
================

The recorder agent writes one session directory per run when
`RECORDER_SESSION_ENABLE=1`.

Session layout
--------------

```
RECORDER_DATASET_DIR/<game>/<session_id>/
  frames/
  actions.jsonl
  sensors.jsonl
  meta.json
  qc.json
```

frames/
- JPEG frames from `RECORDER_FRAME_TOPIC`.
- Filenames look like `<timestamp_ms>_<index>.jpg`.

actions.jsonl
- One JSON object per action:
  `{"timestamp": 1710000000.123, "topic": "act/cmd", "action": {...}}`

sensors.jsonl
- One JSON object per sensor payload:
  `{"timestamp": 1710000000.456, "topic": "scene/state", "payload": {...}}`

meta.json
- `session_id`, `game_id`, `started_at`, `frame_topic`, `sensor_topics`,
  `action_topics`, `profile`, `expected_fps`.

qc.json
- `duration_sec`
- `frames_total`, `frames_dropped`, `dropped_frames_pct`
- `input_lag_ms_mean`, `input_lag_ms_p95`, `input_lag_ms_max`
- `stick_jitter_rms`, `button_spam_rate_hz`
- `frames_written`, `actions_written`, `sensors_written`

Metrics
-------

At shutdown, QC metrics are also published to `metrics/control` as
`recorder.<metric>` with tags `{agent, game_id, session_id}`.

Tools
-----

```
tools/recording_summary.py <session_dir>
```

Configuration
-------------

- `RECORDER_SESSION_ENABLE=1`
- `RECORDER_DATASET_DIR=/mnt/ssd/datasets`
- `RECORDER_FRAME_TOPIC=vision/frame/preview`
- `RECORDER_SENSOR_TOPICS=scene/state,vision/objects,...`
- `RECORDER_SESSION_ID=` (optional)
- `RECORDER_EXPECT_FPS=0` (set to expected FPS to detect drops)
- `RECORDER_GAME_ID=` (optional; falls back to `GAME_ID` or "default")

Legacy samples
--------------

`RECORDER_DIR` still stores `sample_*.json` files for the existing
training pipeline.
