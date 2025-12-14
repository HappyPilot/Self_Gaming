#!/usr/bin/env python3
"""Minimal OCR agent placeholder that forwards OCR text from ocr_easy."""
import json
import os

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
INPUT_TOPIC = os.getenv("OCR_INPUT_TOPIC", "ocr_easy/text")
OUTPUT_TOPIC = os.getenv("OCR_TEXT", "ocr/text")


def on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe(INPUT_TOPIC)
        client.publish("ocr/status", json.dumps({"ok": True, "event": "ocr_stub_ready"}))
    else:
        client.publish("ocr/status", json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))


def on_message(client, _userdata, msg):
    client.publish(OUTPUT_TOPIC, msg.payload)


def main():
    client = mqtt.Client(client_id="ocr_agent_stub", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
