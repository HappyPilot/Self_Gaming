#!/usr/bin/env python3
"""MQTT bridge that translates control events to an HTTP endpoint."""
import json
import logging
import os
import signal
import threading
import urllib.error
import urllib.request

import paho.mqtt.client as mqtt

# --- Setup ---
logging.basicConfig(level=os.getenv("HTTP_BRIDGE_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("http_act_bridge")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
KEY_TOPIC = os.getenv("KEY_TOPIC", "control/keys")
ACT_TOPIC = os.getenv("ACT_TOPIC", "act/cmd")
BRIDGE_URL = os.getenv("CONTROL_BRIDGE_URL") or os.getenv("HTTP_BRIDGE_URL", "http://127.0.0.1:5001/cmd")
REQUEST_TIMEOUT = float(os.getenv("HTTP_BRIDGE_TIMEOUT", "1.5"))

LETTER_KEYS = {chr(code): chr(code) for code in range(ord("a"), ord("z") + 1)}
NUMBER_KEYS = {str(i): str(i) for i in range(10)}
SPECIAL_KEYS = {"enter": "{ENTER}", "return": "{ENTER}", "space": " ", "spacebar": " ", "tab": "{TAB}",
                "escape": "{ESC}", "esc": "{ESC}", "backspace": "{BACKSPACE}", "delete": "{DEL}",
                "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}"}
MODIFIER_KEYS = {"shift": "shift", "ctrl": "control", "control": "control", "alt": "alt",
                 "option": "alt", "cmd": "command", "command": "command"}


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class HttpActBridge:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="http_act_bridge")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _post_payload(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(BRIDGE_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status >= 300:
                    logger.warning("HTTP bridge returned status %s", resp.status)
        except urllib.error.URLError as exc:
            logger.warning("Failed to reach HTTP bridge at %s: %s", BRIDGE_URL, exc)

    def _handle_key_message(self, data: dict):
        key_name = str(data.get("key") or data.get("action") or "").strip().lower()
        if not key_name or key_name.startswith(("click", "mouse", "move")):
            return
        
        key = SPECIAL_KEYS.get(key_name) or LETTER_KEYS.get(key_name) or NUMBER_KEYS.get(key_name)
        if key is None and len(key_name) == 1: key = key_name
        
        if key is None:
            logger.warning("Unsupported key '%s'", key_name)
            return
            
        payload = {"type": "keyboard", "key": key}
        mods = [MODIFIER_KEYS[m.lower()] for m in (data.get("modifiers") or []) if m.lower() in MODIFIER_KEYS]
        if mods: payload["modifiers"] = mods
        
        self._post_payload(payload)
        logger.info("Sent key: %s, mods: %s", key_name, mods)

    def _handle_mouse_click(self, action: str):
        self._post_payload({"type": "mouse", "action": action})
        logger.info("Sent mouse action: %s", action)

    def _handle_mouse_move(self, dx: int, dy: int):
        self._post_payload({"type": "mouse_move", "dx": dx, "dy": dy})
        logger.info("Sent mouse move: dx=%s, dy=%s", dx, dy)

    def _handle_act_message(self, data: dict):
        action = str(data.get("action") or "").lower()
        logger.debug("Processing act message: %s", data)
        if action in {"click_primary", "click_secondary", "click_middle"}:
            self._handle_mouse_click(action)
        elif action == "mouse_move":
            self._handle_mouse_move(int(data.get("dx", 0)), int(data.get("dy", 0)))
        elif action in {"mouse_hold", "mouse_release"}:
            act = "mouse_down" if action == "mouse_hold" else "mouse_up"
            button = data.get("button", "left")
            self._post_payload({"type": "mouse", "action": act, "button": button})
            logger.info("Sent %s for button %s", act, button)
        elif action == "key_press":
            self._handle_key_message(data)

    def _on_connect(self, client, _userdata, _flags, rc):
        rc_int = _as_int(rc)
        if rc_int == 0:
            topics = [t for t in {KEY_TOPIC, ACT_TOPIC} if t]
            for topic in topics: client.subscribe(topic)
            logger.info("Connected to MQTT; bridging topics: %s", topics)
        else:
            logger.error("MQTT connect failed rc=%s", rc_int)

    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0:
            logger.warning("Unexpectedly disconnected from MQTT: rc=%s", _as_int(rc))

    def _on_message(self, cli, _userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", "ignore")
            data = json.loads(payload) if payload.startswith("{") else {"key": payload}
            logger.debug("MQTT %s -> %s", msg.topic, payload)
            if msg.topic == KEY_TOPIC:
                self._handle_key_message(data)
            elif msg.topic == ACT_TOPIC and isinstance(data, dict):
                self._handle_act_message(data)
        except Exception as e:
            logger.error("Error processing message on topic %s: %s", msg.topic, e, exc_info=True)
            
    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("HTTP Action Bridge shut down.")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received, shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = HttpActBridge()
    agent.run()

if __name__ == "__main__":
    main()
