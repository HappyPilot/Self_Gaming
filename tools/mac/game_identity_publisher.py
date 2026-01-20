#!/usr/bin/env python3
"""Publish frontmost app identity to MQTT (window/process, no OCR)."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Optional

import paho.mqtt.client as mqtt

JETSON_IP = os.environ.get("JETSON_IP", "10.0.0.68")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC = os.environ.get("GAME_IDENTITY_TOPIC", "game/identity")
POLL_SEC = float(os.environ.get("GAME_IDENTITY_POLL_SEC", "1.0"))
PUBLISH_SEC = float(os.environ.get("GAME_IDENTITY_PUBLISH_SEC", "5.0"))
LOG_LEVEL = os.environ.get("GAME_IDENTITY_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="[game_identity] %(message)s")
logger = logging.getLogger("game_identity")


def _run_osascript(script: str) -> Optional[str]:
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def _front_app_name() -> Optional[str]:
    return _run_osascript('tell application "System Events" to get name of first application process whose frontmost is true')


def _front_window_title() -> Optional[str]:
    script = (
        'tell application "System Events" to tell (first application process whose frontmost is true) '
        'if (count of windows) > 0 then get name of front window'
    )
    return _run_osascript(script)


def _front_bundle_id() -> Optional[str]:
    return _run_osascript('tell application "System Events" to get bundle identifier of first application process whose frontmost is true')


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_game"


def main() -> None:
    client = mqtt.Client(client_id="game_identity_publisher", protocol=mqtt.MQTTv311)
    connected = {"ok": False}

    def _on_connect(_client, _userdata, _flags, rc):
        connected["ok"] = rc == 0
        logger.info("connected rc=%s", rc)

    def _on_disconnect(_client, _userdata, rc):
        connected["ok"] = False
        logger.warning("disconnected rc=%s", rc)

    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.connect(JETSON_IP, MQTT_PORT, 30)
    client.loop_start()
    logger.info("publishing to %s:%s topic=%s", JETSON_IP, MQTT_PORT, TOPIC)

    last_publish = 0.0
    last_payload = None
    try:
        while True:
            app = _front_app_name()
            title = _front_window_title()
            bundle = _front_bundle_id()
            raw = app or title or bundle or ""
            game_id = _slugify(raw) if raw else "unknown_game"
            payload = {
                "ok": True,
                "timestamp": time.time(),
                "app_name": app,
                "window_title": title,
                "bundle_id": bundle,
                "game_id": game_id,
                "source": "front_app",
            }
            now = time.time()
            if not connected["ok"]:
                try:
                    client.reconnect()
                except Exception:
                    time.sleep(1.0)
                    continue
            if payload != last_payload or (now - last_publish) >= PUBLISH_SEC:
                client.publish(TOPIC, json.dumps(payload))
                last_publish = now
                last_payload = payload
            time.sleep(max(0.2, POLL_SEC))
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
