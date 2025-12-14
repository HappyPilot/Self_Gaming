#!/usr/bin/env python3
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import paho.mqtt.client as mqtt
import logging

logging.basicConfig(level=os.getenv("MEM_LOG_LEVEL", "INFO"))
logger = logging.getLogger("mem_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_QUERY_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query")
MEM_REPLY_TOPIC = os.getenv("MEM_REPLY_TOPIC", "mem/reply")

kv_store = {}
vector_notes = defaultdict(list)
episode_summaries = deque(maxlen=int(os.getenv("MEM_EPISODE_MAX", "500")))
pinned_path = Path(os.getenv("MEM_PINNED_PATH", "/mnt/ssd/memory/pinned.json"))
MAX_PINNED = int(os.getenv("MAX_PINNED_EPISODES", "100"))
pinned_episodes = []


def load_pinned():
    if pinned_path.exists():
        try:
            data = json.loads(pinned_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                pinned_episodes.extend(data)
        except Exception:
            pass


def save_pinned():
    try:
        pinned_path.parent.mkdir(parents=True, exist_ok=True)
        pinned_path.write_text(json.dumps(pinned_episodes), encoding="utf-8")
    except Exception:
        pass


def on_connect(client, userdata, flags, rc):
    topics = [(MEM_STORE_TOPIC, 0), (MEM_QUERY_TOPIC, 0)]
    if rc == 0:
        client.subscribe(topics)
        client.publish(MEM_REPLY_TOPIC, json.dumps({"ok": True, "event": "mem_agent_ready"}))
        load_pinned()
    else:
        client.publish(MEM_REPLY_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))


def store(data):
    op = data.get("op")
    key = data.get("key")
    if op == "set":
        kv_store[key] = data.get("value")
    elif op == "append":
        kv_store.setdefault(key, []).append(data.get("value"))
    elif op == "vector_append":
        vector_notes[key].append(data.get("value"))
    elif op == "episode_summary":
        summary = data.get("value")
        if summary:
            episode_summaries.append({"key": key, "summary": summary})
    elif op == "pin_candidate":
        candidate = data.get("value")
        if candidate:
            insert_pinned(candidate)
    return {"ok": True, "event": "mem_stored", "key": key, "op": op, "timestamp": time.time()}


def insert_pinned(candidate):
    required = {"episode_id", "timestamp", "score", "summary"}
    if not required.issubset(candidate):
        logger = None
    pinned_episodes.append(candidate)
    pinned_episodes.sort(key=lambda e: (-float(e.get("score", 0.0)), -float(e.get("timestamp", 0.0))))
    if len(pinned_episodes) > MAX_PINNED:
        pinned_episodes.pop()
    save_pinned()


def query(data):
    key = data.get("key")
    mode = data.get("mode", "kv")
    if mode == "vector":
        value = vector_notes.get(key, [])
    elif mode == "episodes":
        value = list(episode_summaries)
    elif mode == "pinned":
        limit = int(data.get("limit", 10))
        tag = data.get("tag")
        value = pinned_episodes
        if tag:
            value = [ep for ep in pinned_episodes if tag in (ep.get("tags") or [])]
        value = value[:limit]
    else:
        value = kv_store.get(key)
    return {"ok": True, "event": "mem_result", "key": key, "value": value, "mode": mode}


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}
    if msg.topic == MEM_STORE_TOPIC:
        result = store(data)
    else:
        result = query(data)
    client.publish(MEM_REPLY_TOPIC, json.dumps(result))


def main():
    client = mqtt.Client(client_id="mem_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    try:
        client.loop_forever()
    finally:
        save_pinned()


if __name__ == "__main__":
    main()
