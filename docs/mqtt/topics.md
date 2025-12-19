# MQTT Topics

All payloads are JSON. Many producers include "ok": true and "timestamp", but not every topic guarantees those fields.

## Topic tree (key topics)
- vision/
  - vision/mean: grayscale mean brightness from vision_agent
  - vision/frame: JPEG frame payload (base64)
  - vision/snapshot: on-demand snapshot payload (base64)
  - vision/objects: object detections from object_detection_agent
  - vision/observation: fused perception payload from perception_agent
  - vision/config: runtime config (vision mode, etc)
  - vision/status: vision mode status updates
- ocr/
  - ocr/text: OCR output from ocr_agent
  - ocr_easy/text: OCR output from ocr_easy_agent
- cursor/state: cursor tracker output
- scene/state: fused scene state (mean, text, objects, player, etc)
- teacher/action: teacher_agent suggested action and reasoning
- act/cmd: action command produced by policy_agent or demo stub
- control/keys: legacy key command topic (policy_agent)
- act/result: actuation feedback from act_agent/control bridge
- train/reward: reward events from reward_manager
- train/status: trainer status events
- train/jobs: training jobs published by teacher/train manager
- metrics/latency: latency events (see schemas/latency_event.schema.json)
- logs/summary: log monitor summary
- logs/alerts: log monitor alerts

## Payload examples

### vision/frame
```json
{
  "ok": true,
  "timestamp": 1712345678.1,
  "image_b64": "...",
  "width": 1280,
  "height": 720
}
```

### vision/mean
```json
{
  "ok": true,
  "mean": 0.47,
  "timestamp": 1712345678.1
}
```

### vision/objects
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

Note: box coordinates are in the producer coordinate space (pixel or normalized), depending on the detector backend.

### ocr_easy/text
```json
{
  "ok": true,
  "text": "Play",
  "backend": "paddle",
  "results": [
    {"text": "Play", "conf": 0.92, "box": [0.12, 0.18, 0.15, 0.06]}
  ]
}
```

### scene/state
```json
{
  "ok": true,
  "event": "scene_update",
  "mean": 0.44,
  "text": ["Play"],
  "objects": [{"label": "enemy", "confidence": 0.82, "bbox": [120, 80, 260, 240]}],
  "player": {"label": "player", "confidence": 0.5, "bbox": [400, 200, 520, 480]},
  "timestamp": 1712345678.2
}
```

### act/cmd
```json
{
  "action": "click",
  "button": "left",
  "timestamp": 1712345678.3,
  "source": "policy_agent"
}
```

### teacher/action
```json
{
  "ok": true,
  "action": "click_play",
  "reasoning": "Detected main menu with Play button",
  "timestamp": 1712345678.4
}
```

### train/reward
```json
{
  "ok": true,
  "reward": 0.2,
  "components": {"loot": 0.1, "progress": 0.1},
  "timestamp": 1712345678.5
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
