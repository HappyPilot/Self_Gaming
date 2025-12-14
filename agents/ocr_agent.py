#!/usr/bin/env python3
"""Minimal OCR agent placeholder that forwards OCR text from ocr_easy."""
import json
import os
import signal
import threading
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
INPUT_TOPIC = os.getenv("OCR_INPUT_TOPIC", "ocr_easy/text")
OUTPUT_TOPIC = os.getenv("OCR_TEXT", "ocr/text")

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class OCRAgentStub:
    def __init__(self):
        self.client = mqtt.Client(client_id="ocr_agent_stub", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe(INPUT_TOPIC)
            client.publish("ocr/status", json.dumps({"ok": True, "event": "ocr_stub_ready"}))
        else:
            client.publish("ocr/status", json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_message(self, client, userdata, msg):
        client.publish(OUTPUT_TOPIC, msg.payload)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()

def _handle_signal(signum, frame):
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = OCRAgentStub()
    agent.run()

if __name__ == "__main__":
    main()
