#!/usr/bin/env python3
import json
import os
import time
from collections import deque

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
VISION_MEAN_TOPIC = os.getenv("VISION_MEAN_TOPIC", "vision/mean")
VISION_SNAPSHOT_TOPIC = os.getenv("VISION_SNAPSHOT_TOPIC", "vision/snapshot")
OBJECT_TOPIC = os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
OCR_TEXT_TOPIC = os.getenv("OCR_TEXT_TOPIC", "ocr/text")
OCR_EASY_TOPIC = os.getenv("OCR_EASY_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
WINDOW_SEC = float(os.getenv("SCENE_WINDOW_SEC", "2.0"))

state = {
    "mean": deque(maxlen=10),
    "easy_text": "",
    "simple_text": "",
    "snapshot_ts": 0.0,
    "objects": [],
    "objects_ts": 0.0,
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe([
            (VISION_MEAN_TOPIC, 0),
            (VISION_SNAPSHOT_TOPIC, 0),
            (OBJECT_TOPIC, 0),
            (OCR_TEXT_TOPIC, 0),
            (OCR_EASY_TOPIC, 0),
            (SIMPLE_OCR_TOPIC, 0),
        ])
        client.publish(SCENE_TOPIC, json.dumps({"ok": True, "event": "scene_agent_ready"}))
    else:
        client.publish(SCENE_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))


def _maybe_publish(client):
    now = time.time()
    if now - state["snapshot_ts"] > WINDOW_SEC:
        return
    if not state["mean"]:
        return
    if state["easy_text"]:
        text_payload = [state["easy_text"]]
    elif state["simple_text"]:
        text_payload = [state["simple_text"]]
    else:
        text_payload = []

    payload = {
        "ok": True,
        "event": "scene_update",
        "mean": state["mean"][-1],
        "trend": list(state["mean"]),
        "text": text_payload,
        "objects": state.get("objects", []),
        "objects_ts": state.get("objects_ts", 0.0),
        "timestamp": now,
    }
    client.publish(SCENE_TOPIC, json.dumps(payload))


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    if msg.topic == VISION_MEAN_TOPIC:
        mean = data.get("mean") if isinstance(data, dict) else None
        if mean is not None:
            state["mean"].append(float(mean))
            state["snapshot_ts"] = time.time()
    elif msg.topic == VISION_SNAPSHOT_TOPIC:
        state["snapshot_ts"] = time.time()
    elif msg.topic == OBJECT_TOPIC:
        objects = data.get("objects") if isinstance(data, dict) else None
        if isinstance(objects, list):
            state["objects"] = objects
            state["objects_ts"] = time.time()
    elif msg.topic in (OCR_TEXT_TOPIC, OCR_EASY_TOPIC):
        text = data.get("text") if isinstance(data, dict) else payload
        if text:
            state["easy_text"] = text.strip()
    elif msg.topic == SIMPLE_OCR_TOPIC:
        text = data.get("text") if isinstance(data, dict) else payload
        if text:
            state["simple_text"] = text.strip()

    _maybe_publish(client)


def main():
    client = mqtt.Client(client_id="scene_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
