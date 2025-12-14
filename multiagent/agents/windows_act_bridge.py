#!/usr/bin/env python3
"""MQTT bridge that writes control events to a Windows HTTP bridge."""
import json
import logging
import os
import urllib.error
import urllib.request

import paho.mqtt.client as mqtt

logging.basicConfig(level=os.getenv("WINDOWS_BRIDGE_LOG_LEVEL", "INFO"))
logger = logging.getLogger("windows_act_bridge")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
KEY_TOPIC = os.getenv("KEY_TOPIC", "control/keys")
ACT_TOPIC = os.getenv("ACT_TOPIC", "act/cmd")
WINDOWS_URL = os.getenv("WINDOWS_BRIDGE_URL", "http://10.0.0.40:5000/")
REQUEST_TIMEOUT = float(os.getenv("WINDOWS_BRIDGE_TIMEOUT", "1.5"))

LETTER_KEYS = {chr(code): chr(code) for code in range(ord("a"), ord("z") + 1)}
NUMBER_KEYS = {str(i): str(i) for i in range(0, 10)}
SPECIAL_KEYS = {
    "enter": "{ENTER}",
    "return": "{ENTER}",
    "space": " ",
    "spacebar": " ",
    "tab": "{TAB}",
    "escape": "{ESC}",
    "esc": "{ESC}",
    "backspace": "{BACKSPACE}",
    "delete": "{DEL}",
    "up": "{UP}",
    "down": "{DOWN}",
    "left": "{LEFT}",
    "right": "{RIGHT}",
}
MODIFIER_KEYS = {
    "shift": "shift",
    "ctrl": "control",
    "control": "control",
    "alt": "alt",
    "option": "alt",
    "cmd": "command",
    "command": "command",
}


def post_payload(payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WINDOWS_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status >= 300:
                logger.warning("Windows bridge returned status %s", resp.status)
    except urllib.error.URLError as exc:
        logger.warning("Failed to reach Windows bridge: %s", exc)


def handle_key_message(data: dict):
    key_name = str(data.get("key") or data.get("action") or "").strip().lower()
    if not key_name:
        return
    if key_name.startswith("click") or key_name.startswith("mouse") or key_name.startswith("move"):
        return
    modifiers = data.get("modifiers") or []
    key = SPECIAL_KEYS.get(key_name) or LETTER_KEYS.get(key_name)
    if key is None and key_name in NUMBER_KEYS:
        key = NUMBER_KEYS[key_name]
    if key is None and len(key_name) == 1:
        key = key_name
    if key is None:
        logger.warning("Unsupported key '%s'", key_name)
        return
    payload = {"type": "keyboard", "key": key}
    mods = [MODIFIER_KEYS[m.lower()] for m in modifiers if MODIFIER_KEYS.get(m.lower())]
    if mods:
        payload["modifiers"] = mods
    post_payload(payload)
    logger.info("Sent key %s with mods %s", key_name, mods or [])


def handle_mouse_click(action: str):
    post_payload({"type": "mouse", "action": action})
    logger.info("Sent mouse action %s", action)


def handle_mouse_move(dx: int, dy: int):
    post_payload({"type": "mouse_move", "dx": dx, "dy": dy})
    logger.info("Sent mouse move (%s,%s)", dx, dy)


def handle_act_message(data: dict):
    action = str(data.get("action") or "").lower()
    if action in {"click_primary", "click_secondary", "click_middle"}:
        handle_mouse_click(action)
    elif action == "mouse_move":
        dx = int(data.get("dx", 0))
        dy = int(data.get("dy", 0))
        handle_mouse_move(dx, dy)
    elif action in {"mouse_hold", "mouse_release"}:
        button = data.get("button", "left")
        act = "mouse_down" if action == "mouse_hold" else "mouse_up"
        post_payload({"type": "mouse", "action": act, "button": button})
        logger.info("Sent %s for button %s", act, button)
    elif action == "key_press":
        key = data.get("key")
        if key:
            modifiers = data.get("modifiers") or []
            handle_key_message({"key": key, "modifiers": modifiers})


def main():
    client = mqtt.Client(client_id="windows_act_bridge", protocol=mqtt.MQTTv311)

    def on_connect(cli, _userdata, _flags, rc):
        if rc == 0:
            for topic in filter(None, {KEY_TOPIC, ACT_TOPIC}):
                cli.subscribe(topic)
            logger.info("Connected to MQTT; bridging %s", {KEY_TOPIC, ACT_TOPIC})
        else:
            logger.error("MQTT connect failed rc=%s", rc)

    def on_message(cli, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {"key": payload}
        if msg.topic == KEY_TOPIC:
            handle_key_message(data)
        elif msg.topic == ACT_TOPIC and isinstance(data, dict):
            handle_act_message(data)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Windows bridge stopped")


if __name__ == "__main__":
    main()
