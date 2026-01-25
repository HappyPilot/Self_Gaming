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
FLAGS_TOPIC = os.getenv("SCENE_FLAGS_TOPIC", "scene/flags")
PUBLISH_FLAGS = os.getenv("GAME_IDENTITY_PUBLISH_FLAGS", "0") != "0"
BIND_MODE = os.getenv("GAME_IDENTITY_BIND_MODE", "auto").strip().lower()
BIND_GAME_ID = os.getenv("GAME_IDENTITY_GAME_ID", "").strip()
GRACE_SEC = float(os.getenv("GAME_IDENTITY_GRACE_SEC", "2.0"))
PUBLISH_INTERVAL = float(os.getenv("GAME_IDENTITY_PUBLISH_INTERVAL", "1.0"))
LOG_LEVEL = os.getenv("GAME_IDENTITY_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="[game_identity] %(message)s")
logger = logging.getLogger("game_identity")
stop_event = threading.Event()
active_game_id = _slugify(BIND_GAME_ID) if BIND_GAME_ID else None
last_match_ts = 0.0
last_flags_payload = None
last_flags_publish = 0.0


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


def _match_in_game(game_id: str, now: float) -> bool:
    global active_game_id, last_match_ts
    mode = BIND_MODE if BIND_MODE in {"auto", "strict", "any"} else "auto"
    if mode == "any":
        return game_id != "unknown_game"
    if not active_game_id:
        if mode == "auto" and game_id != "unknown_game":
            active_game_id = game_id
        elif mode == "strict" and BIND_GAME_ID:
            active_game_id = _slugify(BIND_GAME_ID)
    matched = active_game_id is not None and game_id == active_game_id
    if matched:
        last_match_ts = now
        return True
    if last_match_ts and (now - last_match_ts) <= GRACE_SEC:
        return True
    return False


def _maybe_publish_flags(client, payload: dict, now: float) -> None:
    global last_flags_payload, last_flags_publish
    if not PUBLISH_FLAGS or not FLAGS_TOPIC:
        return
    if last_flags_payload == payload and (now - last_flags_publish) < PUBLISH_INTERVAL:
        return
    client.publish(FLAGS_TOPIC, json.dumps({"flags": payload, "timestamp": now}))
    last_flags_payload = payload
    last_flags_publish = now


def _on_message(client, _userdata, msg):
    global active_game_id
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    raw_id = payload.get("game_id") or payload.get("app_name") or payload.get("window_title") or payload.get("bundle_id")
    game_id = _slugify(raw_id) if raw_id else "unknown_game"
    now = time.time()
    value = {
        "game_id": game_id,
        "app_name": payload.get("app_name"),
        "window_title": payload.get("window_title"),
        "bundle_id": payload.get("bundle_id"),
        "source": payload.get("source") or "front_app",
        "timestamp": payload.get("timestamp") or now,
        "received_at": now,
    }
    client.publish(MEM_STORE_TOPIC, json.dumps({"op": "set", "key": "game_identity", "value": value}))
    in_game = _match_in_game(game_id, now)
    flags_payload = {
        "in_game": in_game,
        "game_id": active_game_id or game_id,
        "front_app": payload.get("app_name"),
        "window_title": payload.get("window_title"),
        "source": value.get("source"),
    }
    _maybe_publish_flags(client, flags_payload, now)


def main() -> None:
    client = mqtt.Client(client_id="game_identity_agent", protocol=mqtt.MQTTv311)
    client.on_connect = _on_connect
    client.on_message = _on_message
    if PUBLISH_FLAGS and BIND_MODE == "strict" and not BIND_GAME_ID:
        logger.warning("GAME_IDENTITY_BIND_MODE=strict but GAME_IDENTITY_GAME_ID is empty; in_game will stay false")
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_start()
    stop_event.wait()
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
