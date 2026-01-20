#!/usr/bin/env python3
"""Persist game identity info from the host (front app/window)."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
IDENTITY_TOPIC = os.getenv("GAME_IDENTITY_TOPIC", "game/identity")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
LOG_LEVEL = os.getenv("GAME_IDENTITY_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="[game_identity] %(message)s")
logger = logging.getLogger("game_identity")
stop_event = threading.Event()


def _slugify(value: str) -> str:
    lowered = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_game"


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


def _on_connect(client, _userdata, _flags, rc):
    if _as_int(rc) == 0:
        client.subscribe([(IDENTITY_TOPIC, 0)])
        logger.info("Connected; subscribed to %s", IDENTITY_TOPIC)
    else:
        logger.error("Connect failed: %s", rc)


def _on_message(client, _userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    raw_id = payload.get("game_id") or payload.get("app_name") or payload.get("window_title") or payload.get("bundle_id")
    game_id = _slugify(raw_id) if raw_id else "unknown_game"
    value = {
        "game_id": game_id,
        "app_name": payload.get("app_name"),
        "window_title": payload.get("window_title"),
        "bundle_id": payload.get("bundle_id"),
        "source": payload.get("source") or "front_app",
        "timestamp": payload.get("timestamp") or time.time(),
        "received_at": time.time(),
    }
    client.publish(MEM_STORE_TOPIC, json.dumps({"op": "set", "key": "game_identity", "value": value}))


def main() -> None:
    client = mqtt.Client(client_id="game_identity_agent", protocol=mqtt.MQTTv311)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_start()
    stop_event.wait()
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
