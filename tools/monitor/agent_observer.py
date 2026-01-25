#!/usr/bin/env python3
"""Periodic snapshot + action capture for agent behavior review."""
from __future__ import annotations

import base64
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

SNAPSHOT_CMD_TOPIC = os.getenv("SNAPSHOT_CMD_TOPIC", "vision/cmd")
SNAPSHOT_TOPIC = os.getenv("SNAPSHOT_TOPIC", "vision/snapshot")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACTION_TOPICS = [t.strip() for t in os.getenv("ACTION_TOPICS", "act/cmd,act/result,policy_brain/cmd,teacher/action").split(",") if t.strip()]

SAMPLE_COUNT = int(os.getenv("SAMPLE_COUNT", "10"))
SAMPLE_INTERVAL_SEC = float(os.getenv("SAMPLE_INTERVAL_SEC", "600"))
ACTION_WINDOW_SEC = float(os.getenv("ACTION_WINDOW_SEC", "60"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/mnt/ssd/logs/agent_observer"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

action_log: Deque[Dict[str, object]] = deque(maxlen=5000)
snapshot_payload: Optional[Dict[str, object]] = None
latest_scene: Dict[str, object] = {}


def _now() -> float:
    return time.time()


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        topics: List[Tuple[str, int]] = [(SNAPSHOT_TOPIC, 0), (SCENE_TOPIC, 0)]
        topics += [(topic, 0) for topic in ACTION_TOPICS]
        client.subscribe(topics)
    else:
        raise RuntimeError(f"MQTT connect failed: {rc}")


def _on_message(_client, _userdata, msg):
    global snapshot_payload, latest_scene
    payload = msg.payload.decode("utf-8", "ignore")
    if msg.topic == SNAPSHOT_TOPIC:
        try:
            snapshot_payload = json.loads(payload)
        except Exception:
            snapshot_payload = None
        return
    if msg.topic == SCENE_TOPIC:
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                latest_scene = data
        except Exception:
            return
        return
    # Action topics
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}
    action_log.append({"ts": _now(), "topic": msg.topic, "payload": data})


def _request_snapshot(client) -> Optional[Dict[str, object]]:
    global snapshot_payload
    snapshot_payload = None
    client.publish(SNAPSHOT_CMD_TOPIC, json.dumps({"cmd": "snapshot"}))
    deadline = _now() + 5.0
    while _now() < deadline:
        if snapshot_payload:
            return snapshot_payload
        time.sleep(0.1)
    return None


def _save_snapshot(payload: Dict[str, object], index: int) -> Optional[str]:
    if not payload or not payload.get("ok"):
        return None
    image_b64 = payload.get("image_b64")
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        return None
    filename = OUTPUT_DIR / f"snapshot_{index:02d}_{int(_now())}.jpg"
    try:
        filename.write_bytes(raw)
    except Exception:
        return None
    return str(filename)


def _actions_in_window(center_ts: float, window_sec: float) -> List[Dict[str, object]]:
    half = max(1.0, window_sec / 2.0)
    start = center_ts - half
    end = center_ts + half
    return [entry for entry in list(action_log) if start <= entry.get("ts", 0) <= end]


def main() -> None:
    client = mqtt.Client(client_id="agent_observer", protocol=mqtt.MQTTv311)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_start()

    samples: List[Dict[str, object]] = []
    start_time = _now()
    for idx in range(SAMPLE_COUNT):
        target_ts = start_time + idx * SAMPLE_INTERVAL_SEC
        while _now() < target_ts:
            time.sleep(0.5)
        snapshot = _request_snapshot(client)
        snap_path = _save_snapshot(snapshot, idx) if snapshot else None
        # wait for post-window actions
        time.sleep(max(1.0, ACTION_WINDOW_SEC / 2.0))
        actions = _actions_in_window(target_ts, ACTION_WINDOW_SEC)
        sample = {
            "index": idx,
            "timestamp": target_ts,
            "snapshot_path": snap_path,
            "snapshot_ok": bool(snapshot and snapshot.get("ok")),
            "actions": actions,
            "action_count": len(actions),
            "scene": latest_scene,
        }
        samples.append(sample)
        (OUTPUT_DIR / "samples.json").write_text(json.dumps(samples, indent=2))

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
