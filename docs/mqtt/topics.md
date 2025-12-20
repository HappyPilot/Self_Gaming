# MQTT Topics

All payloads are JSON.

## Conventions
- ok: boolean success flag. When ok is false, an "error" string may be present.
- timestamp: Unix seconds (float). Required on some topics only.
- Bounding boxes:
  - vision/objects uses "box" in pixel XYXY format: [x1, y1, x2, y2].
  - vision/observation and scene/state use "bbox" in normalized XYXY format: [x1, y1, x2, y2] with values in [0, 1].
  - ocr_easy/text results use "box" in normalized XYWH format: [x, y, w, h] with values in [0, 1].

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
- logs/summary: log monitor summary
- logs/alerts: log monitor alerts

## Payload examples

### vision/frame
Required: ok, timestamp, image_b64, width, height
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
```json
{
  "action": "click",
  "button": "left",
  "timestamp": 1712345678.3,
  "source": "policy_agent"
}
```

### teacher/action
Required: ok, text, timestamp
```json
{
  "ok": true,
  "text": "Click the Play button",
  "timestamp": 1712345678.4,
  "game_id": "poe"
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
