#!/usr/bin/env python3
import json
import os
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACT_ALIAS = os.getenv("ACT_CMD_ALIAS", "act/request")
TEACHER_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
REPLAY_TOPIC = os.getenv("REPLAY_STORE_TOPIC", "replay/store")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_SUMMARY_KEY = os.getenv("MEM_EPISODE_KEY", "episodes")
DATA_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
MAX_RECORDS = int(os.getenv("RECORDER_MAX", "1000"))
PIN_THRESHOLD = float(os.getenv("PIN_THRESHOLD", "0.0"))

DATA_DIR.mkdir(parents=True, exist_ok=True)

scene_cache = deque(maxlen=10)
record_count = 0
last_teacher_action = {"source": None, "action": None, "timestamp": 0.0}
last_reward = {"value": 0.0, "timestamp": 0.0}


def persist(record):
    global record_count
    record_count += 1
    ts = record["timestamp"]
    path = DATA_DIR / f"sample_{int(ts)}_{record_count:05d}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f)
    return path


def publish_transition(client, record):
    if REPLAY_TOPIC:
        transition = {
            "timestamp": record.get("timestamp"),
            "state": record.get("scene"),
            "next_state": None,
            "action": record.get("action"),
            "reward": record.get("reward"),
            "teacher": record.get("teacher"),
            "priority": abs(record.get("reward", {}).get("value", 0.0)) + 1e-3,
        }
        client.publish(REPLAY_TOPIC, json.dumps(transition))


def publish_episode_summary(client, record):
    if not MEM_STORE_TOPIC:
        return
    scene = record.get("scene", {})
    summary = {
        "timestamp": record.get("timestamp"),
        "goal_id": scene.get("goal_id"),
        "task_id": scene.get("task_id"),
        "action": record.get("action"),
        "teacher": record.get("teacher"),
        "reward": record.get("reward"),
        "text": scene.get("text"),
    }
    payload = {"op": "episode_summary", "key": MEM_SUMMARY_KEY, "value": summary}
    client.publish(MEM_STORE_TOPIC, json.dumps(payload))


def compute_score(scene, reward):
    score = float(reward.get("value", 0.0))
    details = reward.get("details") or {}
    if details.get("success"):
        score += 1.0
    if scene.get("goal_status") == "success":
        score += 1.0
    return score


def maybe_pin_episode(client, record, path):
    if not MEM_STORE_TOPIC:
        return
    scene = record.get("scene", {})
    reward = record.get("reward", {})
    score = compute_score(scene, reward)
    if score < PIN_THRESHOLD:
        return
    tags = []
    goal_id = scene.get("goal_id")
    if goal_id:
        tags.append(f"goal:{goal_id}")
    if scene.get("mode") == "sim":
        tags.append("sim")
    elif scene.get("mode") == "live":
        tags.append("live")
    payload = {
        "op": "pin_candidate",
        "key": "pinned",
        "value": {
            "episode_id": path.name,
            "timestamp": record.get("timestamp"),
            "score": score,
            "summary": {
                "goal_id": goal_id,
                "task_id": scene.get("task_id"),
                "reward": reward,
                "teacher": record.get("teacher"),
            },
            "payload_path": str(path),
            "tags": tags,
        },
    }
    client.publish(MEM_STORE_TOPIC, json.dumps(payload))


def on_connect(client, userdata, flags, rc):
    topics = {(SCENE_TOPIC, 0), (ACT_TOPIC, 0)}
    if TEACHER_TOPIC:
        topics.add((TEACHER_TOPIC, 0))
    if REWARD_TOPIC:
        topics.add((REWARD_TOPIC, 0))
    if ACT_ALIAS and ACT_ALIAS != ACT_TOPIC:
        topics.add((ACT_ALIAS, 0))
    if rc == 0:
        client.subscribe(list(topics))
        client.publish(
            "recorder/status",
            json.dumps({"ok": True, "event": "recorder_ready", "path": str(DATA_DIR)}),
        )
    else:
        client.publish(
            "recorder/status",
            json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}),
        )


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    global last_teacher_action, last_reward

    if msg.topic == SCENE_TOPIC and data.get("ok"):
        scene_cache.append((time.time(), data))
    elif msg.topic == TEACHER_TOPIC:
        action_text = data.get("action") if isinstance(data, dict) else payload
        last_teacher_action = {
            "source": data.get("source") if isinstance(data, dict) else None,
            "action": action_text,
            "timestamp": time.time(),
            "reasoning": data.get("reasoning") if isinstance(data, dict) else None,
        }
    elif msg.topic == REWARD_TOPIC:
        value = data.get("reward") if isinstance(data, dict) else None
        try:
            reward_value = float(value)
        except (TypeError, ValueError):
            reward_value = 0.0
        last_reward = {"value": reward_value, "timestamp": time.time(), "details": data}
    elif msg.topic in (ACT_TOPIC, ACT_ALIAS):
        if not scene_cache:
            return
        ts, scene = scene_cache[-1]
        record = {
            "timestamp": time.time(),
            "scene_time": ts,
            "scene": scene,
            "action": data,
            "teacher": last_teacher_action,
            "reward": last_reward,
        }
        episode_path = persist(record)
        publish_transition(client, record)
        publish_episode_summary(client, record)
        maybe_pin_episode(client, record, episode_path)
        if record_count > MAX_RECORDS:
            oldest = sorted(DATA_DIR.glob("sample_*.json"))
            for path in oldest[: record_count - MAX_RECORDS]:
                try:
                    path.unlink()
                except Exception:
                    pass
            client.publish(
                "recorder/status",
                json.dumps({"ok": True, "event": "pruned", "count": record_count}),
            )


def main():
    client = mqtt.Client(client_id="recorder_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
