# MQTT Topics

All payloads are JSON.

## Conventions
- ok: boolean success flag. When ok is false, an "error" string may be present.
- timestamp: Unix seconds (float). Required on some topics only.
- Bounding boxes:
  - vision/objects uses "box" in pixel XYXY format: [x1, y1, x2, y2].
  - vision/observation and scene/state use "bbox" in normalized XYXY format: [x1, y1, x2, y2] with values in [0, 1].
  - ocr_easy/text results use "box" in normalized XYWH format: [x, y, w, h] with values in [0, 1].
- Object labels:
  - vision/objects uses "class" for the detector label.
  - scene/state uses "label"; scene_agent maps class -> label when fusing.
- Control metrics:
  - metrics/control uses "metric" and "value" fields.
  - expected metric names: control/tick_ms, control/chunk_boundary_jerk, control/next_chunk_ready_ratio.

## Topic tree (key topics)
- vision/
  - vision/mean: grayscale mean brightness from vision_agent
  - vision/frame/preview: JPEG preview frames (base64, low quality)
  - vision/frame/full: JPEG full frames (base64, high quality)
  - vision/frame: legacy alias (publish by setting VISION_FRAME_PREVIEW_TOPIC=vision/frame)
  - vision/snapshot: on-demand snapshot payload (base64)
  - vision/objects: object detections from object_detection_agent
  - vision/observation: fused perception payload from perception_agent
  - vision/embeddings: visual embeddings from vl_jepa_agent
  - vision/config: runtime config (vision mode, etc)
  - vision/status: vision mode status updates
- ocr/
  - ocr/text: OCR output from ocr_agent
  - ocr_easy/text: OCR output from ocr_easy_agent
- cursor/state: cursor tracker output
- scene/state: fused scene state (mean, text, objects, player, etc)
- teacher/action: teacher_agent guidance text
- act/cmd: action command produced by policy_agent or demo stub
- control/keys: legacy key command topic (policy_agent)
- act/result: actuation feedback from act_agent/control bridge
- train/reward: reward events from reward_manager
- train/status: trainer status events
- train/jobs: training jobs published by teacher/train manager
- metrics/latency: latency events (see schemas/latency_event.schema.json)
- metrics/control: control loop metrics (see schemas/control_metric.schema.json)
- logs/summary: log monitor summary
- logs/alerts: log monitor alerts
- progress/status: progress agent summary/status
- progress/understanding: progress agent understanding metrics (location memory + grounding)

## Payload examples

### vision/frame/preview
Required: ok, timestamp, width, height, variant, image_b64 (or shm descriptor when FRAME_TRANSPORT=shm)
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "image_b64": "...",
  "width": 1280,
  "height": 720,
  "variant": "preview"
}
```

If FRAME_TRANSPORT=shm, publish a descriptor instead of image_b64:
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "transport": "shm",
  "encoding": "jpeg",
  "shm_name": "sg_frame_1234_abcd_0",
  "shm_size": 235421,
  "shm_seq": 42,
  "width": 1280,
  "height": 720,
  "variant": "preview"
}
```
Note: SHM transport requires a shared IPC namespace for vision_agent and all consumers (ipc: host or ipc: service:mq) and enough shm_size.

### vision/frame/full
Required: ok, timestamp, width, height, variant, image_b64 (or shm descriptor when FRAME_TRANSPORT=shm)
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "image_b64": "...",
  "width": 1280,
  "height": 720,
  "variant": "full"
}
```

### vision/mean
Required: ok, mean, timestamp
```json
{
  "ok": true,
  "mean": 0.47,
  "timestamp": 1712345678.1
}
```

### vision/objects
Required: ok, timestamp, objects
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "frame_ts": 1712345677.9,
  "objects": [
    {"class": "enemy", "confidence": 0.82, "box": [120, 80, 260, 240]}
  ]
}
```

Note: object_detection_agent publishes pixel boxes in XYXY format.

### vision/observation
Required: ok, timestamp, frame_id
Schema: `schemas/observation.schema.json`
```json
{
  "ok": true,
  "frame_id": 120,
  "timestamp": 1712345678.1,
  "yolo_objects": [
    {"label": "enemy", "confidence": 0.82, "bbox": [0.12, 0.08, 0.26, 0.24], "extra": {}}
  ],
  "text_zones": {
    "dialog": {"text": "Play", "confidence": 0.92, "bbox": [0.12, 0.18, 0.24, 0.28]}
  },
  "player_candidate": {"label": "player", "confidence": 0.5, "bbox": [0.4, 0.2, 0.52, 0.48]}
}
```
Note: bbox values are normalized XYXY in [0, 1].

### vision/embeddings
Required: ok, timestamp, embedding, dim
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "frame_ts": 1712345677.9,
  "embedding": [0.01, -0.02, 0.03],
  "dim": 512,
  "backend": "torchscript"
}
```
Note: backend is typically `torchscript`, `tensorrt`, `siglip2`, or `dummy` depending on vl_jepa_agent config.

### ocr_easy/text
Required: ok, text, results, backend
```json
{
  "ok": true,
  "text": "Play",
  "backend": "easyocr",
  "results": [
    {"text": "Play", "conf": 0.92, "box": [0.12, 0.18, 0.15, 0.06]}
  ]
}
```

Note: backend is "paddle" or "easyocr" depending on OCR_BACKEND.

### scene/state
Required: ok, event, mean, text, objects, timestamp
```json
{
  "ok": true,
  "event": "scene_update",
  "mean": 0.44,
  "text": ["Play"],
  "objects": [{"label": "enemy", "confidence": 0.82, "bbox": [0.12, 0.08, 0.26, 0.24]}],
  "player": {"label": "player", "confidence": 0.5, "bbox": [0.4, 0.2, 0.52, 0.48]},
  "timestamp": 1712345678.2
}
```

Note: when scene/state is built from vision/observation (perception_agent), bbox values are normalized.

### act/cmd
Required: action
Schema: `schemas/action.schema.json`
```json
{
  "action": "click",
  "button": "left",
  "timestamp": 1712345678.3,
  "source": "policy_agent"
}
```

### teacher/action
Required: ok, action, reasoning, timestamp
```json
{
  "ok": true,
  "action": "Click the Play button",
  "reasoning": "The Play button is highlighted and expected to continue.",
  "text": "Click the Play button",
  "timestamp": 1712345678.4,
  "rules_used": 1,
  "recent_critical_used": 0,
  "game_id": "poe",
  "context_game": "Path of Exile"
}
```

### train/reward
Required: ok, reward, timestamp
```json
{
  "ok": true,
  "reward": 0.2,
  "components": {"loot": 0.1, "progress": 0.1},
  "timestamp": 1712345678.5
}
```

### progress/status
High-level health snapshot from progress_agent.
```json
{
  "ok": true,
  "event": "progress_status",
  "timestamp": 1712345678.6,
  "status": "OK",
  "total_reward": 12.34,
  "last_reward_age": 8,
  "scene_age": 1,
  "understanding": { "...": "see progress/understanding" }
}
```

### progress/understanding
Location memory + grounding metrics (game-agnostic).
```json
{
  "ok": true,
  "event": "understanding_update",
  "timestamp": 1712345678.7,
  "understanding": {
    "locations": {
      "count": 12,
      "unique": 12,
      "assignments": 340,
      "new_rate": 0.18,
      "revisit_rate": 0.82,
      "current_id": 7,
      "current_similarity": 0.92,
      "current_age_sec": 14,
      "transitions": 36,
      "top": [{"id": 3, "visits": 120, "last_seen_sec": 4}]
    },
    "embeddings": {
      "delta_last": 0.0123,
      "delta_avg": 0.0345
    },
    "objects": {
      "vocab_size": 58,
      "new_rate": 0.07,
      "last_new": ["monster", "altar"],
      "last_count": 6
    },
    "ocr": {
      "vocab_size": 140,
      "new_rate": 0.05,
      "last_new": ["victory"],
      "last_count": 12
    },
    "grounding": {
      "clicks": 430,
      "hits": 140,
      "hit_rate": 0.325,
      "targeted_clicks": 210,
      "cursor_clicks": 45,
      "clicks_with_targets": 180,
      "clicks_with_objects": 260,
      "hits_on_targets": 90,
      "hits_on_objects": 50,
      "target_hit_rate": 0.5,
      "object_hit_rate": 0.192,
      "last_hit": true,
      "last_reason": "bbox"
    }
  }
}
```

### metrics/latency
```json
{
  "event": "latency",
  "stage": "detect",
  "duration_ms": 42.3,
  "sla_ms": 100,
  "timestamp": 1712345678.9,
  "tags": {"agent": "object_detection_agent", "frame_id": 120}
}
```
Schema: `schemas/latency_event.schema.json`

### metrics/control
Required: event, metric, value, timestamp
```json
{
  "event": "control_metric",
  "metric": "control/tick_ms",
  "value": 12.3,
  "ok": true,
  "timestamp": 1712345678.9,
  "tags": {"window": 50}
}
```
Schema: `schemas/control_metric.schema.json`

Note: tick_ms and next_chunk_ready_ratio may be sampled (CONTROL_METRIC_SAMPLE_EVERY) and use a rolling window size (CONTROL_READY_WINDOW).
