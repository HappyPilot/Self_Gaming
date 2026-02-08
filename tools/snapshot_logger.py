#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("YOLO_LOG_TOPIC", "vision/objects")
LOG_DIR = Path(os.getenv("YOLO_LOG_DIR", "/mnt/ssd/logs/yolo"))
LOG_FILE = os.getenv("YOLO_LOG_FILE", "").strip()
LOG_DIR.mkdir(parents=True, exist_ok=True)

client = mqtt.Client()

if not LOG_FILE:
    safe_topic = TOPIC.strip().replace("/", "_").replace(":", "_")
    if not safe_topic:
        safe_topic = "vision_objects"
    LOG_FILE = f"{safe_topic}.log"
outfile = open(LOG_DIR / LOG_FILE, "a", encoding="utf-8")

def on_message(_cli, _userdata, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    payload = msg.payload.decode("utf-8", "ignore")
    outfile.write(f"[{ts}] {msg.topic}: {payload}\n")
    outfile.flush()

client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, 30)
client.subscribe(TOPIC)
client.loop_forever()
