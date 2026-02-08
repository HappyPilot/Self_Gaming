# Game Schema Retain + 30-Min Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `game/schema` retained so late subscribers always receive the latest profile, then run a 30-minute monitoring pass with screenshots on Jetson.

**Architecture:** Add MQTT `retain=True` to onboarding’s schema publish and verify via a focused unit test. Use a lightweight, one-off monitor script on Jetson to correlate `act/cmd` with `scene/state` and sample screenshots from `vision/frame/preview`.

**Tech Stack:** Python, paho-mqtt, pytest, mosquitto_sub, existing `utils.frame_transport`.

---

### Task 1: Add failing test for retained schema publish

**Files:**
- Create: `agents/tests/test_game_onboarding_agent.py`

**Step 1: Write the failing test**
```python
import json

from agents import game_onboarding_agent as goa


class DummyClient:
    def __init__(self):
        self.calls = []

    def publish(self, topic, payload, retain=False):
        self.calls.append((topic, payload, retain))


def test_schema_publish_retained():
    agent = goa.GameOnboardingAgent()
    agent.client = DummyClient()
    schema = {
        "game_id": "test_game",
        "ui_layout": {},
        "controls": {},
        "signals": {},
        "profile": {},
        "profile_status": "ok",
        "llm_status": "ok",
        "notes": [],
    }
    agent._publish_schema(schema)
    retained = [call for call in agent.client.calls if call[0] == goa.SCHEMA_TOPIC]
    assert retained, "Schema publish missing"
    assert retained[0][2] is True, "Schema publish must be retained"
```

**Step 2: Run test to verify it fails**
Run: `pytest agents/tests/test_game_onboarding_agent.py::test_schema_publish_retained -v`
Expected: FAIL with retain flag `False`.

---

### Task 2: Implement retain publish

**Files:**
- Modify: `agents/game_onboarding_agent.py:441-444`

**Step 1: Write minimal implementation**
```python
payload = {"ok": True, "schema": schema, "timestamp": time.time()}
self.client.publish(SCHEMA_TOPIC, json.dumps(payload), retain=True)
```

**Step 2: Run test to verify it passes**
Run: `pytest agents/tests/test_game_onboarding_agent.py::test_schema_publish_retained -v`
Expected: PASS.

**Step 3: Commit**
```bash
git add agents/game_onboarding_agent.py agents/tests/test_game_onboarding_agent.py
git commit -m "Retain game schema publish"
```

---

### Task 3: Verify retained schema via MQTT (Jetson)

**Files:**
- None (ops only)

**Step 1: Deploy/checkout branch on Jetson**
Run: `ssh dima@10.0.0.68 "cd ~/self-gaming && git fetch && git checkout game-schema-retain-monitor && git pull"`

**Step 2: Restart onboarding agent**
Run: `ssh dima@10.0.0.68 "cd ~/self-gaming && ./scripts/restart_onboarding.sh"` (adjust to actual process manager)

**Step 3: Verify retained message**
Run: `ssh dima@10.0.0.68 "mosquitto_sub -t game/schema -C 1 -v"`
Expected: immediate output with JSON payload.

---

### Task 4: 30-minute monitor + screenshots (Jetson)

**Files:**
- Create: `/tmp/monitor_game_actions.py`
- Outputs: `/mnt/ssd/logs/monitor_<timestamp>/{events.jsonl,summary.json,shots/*.jpg}`

**Step 1: Write monitor script**
```python
#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime

import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes

HOST = os.getenv("MQTT_HOST", "127.0.0.1")
PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
DURATION_SEC = int(os.getenv("MONITOR_DURATION_SEC", "1800"))
SHOT_EVERY_SEC = int(os.getenv("MONITOR_SHOT_EVERY_SEC", "45"))

stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
base_dir = f"/mnt/ssd/logs/monitor_{stamp}"
shots_dir = os.path.join(base_dir, "shots")
os.makedirs(shots_dir, exist_ok=True)

events_path = os.path.join(base_dir, "events.jsonl")
latest_scene = {}
last_shot = 0.0
counts = {
    "actions": 0,
    "key_press": 0,
    "clicks": 0,
    "enemy_hit": 0,
    "ocr_target": 0,
    "scene_missing": 0,
    "frame_errors": 0,
}


def inside_boxes(target, boxes):
    if not target or not boxes:
        return False
    x = target.get("x")
    y = target.get("y")
    if x is None or y is None:
        return False
    for box in boxes:
        if not isinstance(box, dict):
            continue
        x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def on_message(_client, _userdata, msg):
    global latest_scene
    now = time.time()
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return

    if topic == SCENE_TOPIC and isinstance(payload, dict):
        latest_scene = payload
        return

    if topic == ACT_TOPIC and isinstance(payload, dict):
        counts["actions"] += 1
        action_type = payload.get("action")
        if action_type == "key_press":
            counts["key_press"] += 1
        if action_type in {"click_primary", "click_secondary", "click_middle"}:
            counts["clicks"] += 1

        scene = latest_scene if isinstance(latest_scene, dict) else {}
        enemy_hit = False
        ocr_hit = False
        if not scene:
            counts["scene_missing"] += 1
        else:
            target = payload.get("target_norm") or {}
            enemy_boxes = (scene.get("enemy_bars") or {}).get("boxes") or []
            target_boxes = (scene.get("targets") or {}).get("boxes") or []
            enemy_hit = inside_boxes(target, enemy_boxes)
            ocr_hit = inside_boxes(target, target_boxes)
            if enemy_hit:
                counts["enemy_hit"] += 1
            if ocr_hit:
                counts["ocr_target"] += 1

        event = {
            "ts": now,
            "action": action_type,
            "target_norm": payload.get("target_norm"),
            "enemy_hit": enemy_hit,
            "ocr_target": ocr_hit,
        }
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "
")


def on_frame(_client, _userdata, msg):
    global last_shot
    now = time.time()
    if now - last_shot < SHOT_EVERY_SEC:
        return
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return
    data = get_frame_bytes(payload)
    if not data:
        counts["frame_errors"] += 1
        return
    last_shot = now
    filename = os.path.join(shots_dir, f"{int(now)}.jpg")
    try:
        with open(filename, "wb") as handle:
            handle.write(data)
    except Exception:
        counts["frame_errors"] += 1


client = mqtt.Client(client_id=f"monitor_{stamp}")
client.on_message = on_message
client.message_callback_add(FRAME_TOPIC, on_frame)
client.connect(HOST, PORT, 60)
client.subscribe([(SCENE_TOPIC, 0), (ACT_TOPIC, 0), (FRAME_TOPIC, 0)])

start = time.time()
client.loop_start()
while time.time() - start < DURATION_SEC:
    time.sleep(1)
client.loop_stop()
client.disconnect()

summary = {
    "started_at": start,
    "ended_at": time.time(),
    "duration_sec": DURATION_SEC,
    **counts,
}
if counts["actions"]:
    summary["key_press_rate"] = counts["key_press"] / counts["actions"]
if counts["clicks"]:
    summary["enemy_hit_rate"] = counts["enemy_hit"] / counts["clicks"]
    summary["ocr_target_rate"] = counts["ocr_target"] / counts["clicks"]

with open(os.path.join(base_dir, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
```

**Step 2: Dry run (60s)**
Run: `ssh dima@10.0.0.68 "python3 /tmp/monitor_game_actions.py"` with `MONITOR_DURATION_SEC=60`
Expected: summary.json + at least 1 screenshot.

**Step 3: Full 30-min run**
Run: `ssh dima@10.0.0.68 "MONITOR_DURATION_SEC=1800 python3 /tmp/monitor_game_actions.py"`
Expected: summary.json + ~40 screenshots.

**Step 4: Review results**
Inspect: `/mnt/ssd/logs/monitor_<timestamp>/summary.json` and sampled `shots/*.jpg`.

