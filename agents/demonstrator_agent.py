#!/usr/bin/env python3
"""Heuristic demonstrator that emits simple actions from scene summaries.

This agent watches `scene/state` updates and generates synthetic actions to seed
behavior-cloning data. It publishes commands on both `act/cmd` and its alias so
recorder/training flows receive labeled examples without manual input.
"""
import json
import os
import random
import re
import signal
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import paho.mqtt.client as mqtt

AGENT_NAME = os.getenv("DEMO_AGENT_NAME", "auto_demonstrator")
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACT_CMD_ALIAS = os.getenv("ACT_CMD_ALIAS", "act/request")
CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "control/keys")
ACTION_INTERVAL = float(os.getenv("ACTION_INTERVAL", "0.75"))
MAX_REPEAT = int(os.getenv("ACTION_MAX_REPEAT", "3"))
RULES_ENV = os.getenv("DEMO_RULES", "")
MOUSE_RANGE = int(os.getenv("DEMO_MOUSE_RANGE", "80"))
MIN_MOUSE_DELTA = int(os.getenv("DEMO_MOUSE_MIN_DELTA", "10"))
DEMO_REQUIRE_IN_GAME = os.getenv("DEMO_REQUIRE_IN_GAME", "0") != "0"
DEMO_ALLOW_KEYS = os.getenv("DEMO_ALLOW_KEYS", "0") != "0"

stop_event = threading.Event()


def _as_int(code):
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except Exception:
        return 0


@dataclass
class Rule:
    action: str
    keywords: List[str]
    pattern: Optional[re.Pattern]
    min_mean: Optional[float] = None
    max_mean: Optional[float] = None

    def matches(self, scene_text: str, mean: Optional[float]) -> bool:
        if self.min_mean is not None and (mean is None or mean < self.min_mean):
            return False
        if self.max_mean is not None and (mean is None or mean > self.max_mean):
            return False
        if self.pattern and self.pattern.search(scene_text):
            return True
        if self.keywords:
            lowered = scene_text.lower()
            return any(keyword in lowered for keyword in self.keywords)
        return False


def _load_rules() -> List[Rule]:
    default = [
        Rule(action="press_enter", keywords=["menu", "start", "continue", "dialog", "ok"], pattern=None),
        Rule(action="click_primary", keywords=["button", "click", "play"], pattern=None),
        Rule(action="focus_field", keywords=["input", "field", "enter your"], pattern=None),
        Rule(action="scroll_down", keywords=["list", "scroll"], pattern=None),
        Rule(action="pause", keywords=["loading", "please wait"], pattern=None),
    ]
    if not RULES_ENV:
        return default
    try:
        parsed = json.loads(RULES_ENV)
    except Exception:
        return default
    custom: List[Rule] = []
    if not isinstance(parsed, list):
        return default
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action")
        if not action:
            continue
        keywords = entry.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        pattern_text = entry.get("pattern")
        pattern = None
        if pattern_text:
            try:
                pattern = re.compile(pattern_text, re.IGNORECASE)
            except re.error:
                pattern = None
        min_mean = entry.get("min_mean")
        max_mean = entry.get("max_mean")
        custom.append(
            Rule(
                action=str(action),
                keywords=[str(k).lower() for k in keywords],
                pattern=pattern,
                min_mean=float(min_mean) if isinstance(min_mean, (int, float)) else None,
                max_mean=float(max_mean) if isinstance(max_mean, (int, float)) else None,
            )
        )
    return custom or default


RULES = _load_rules()


class Demonstrator:
    def __init__(self):
        self.lock = threading.Lock()
        self.client = mqtt.Client(client_id=AGENT_NAME, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.client.user_data_set({})

        self.last_action = None
        self.last_action_ts = 0.0
        self.repeat_count = 0
        self.rng = random.Random()

    # MQTT callbacks --------------------------------------------------
    def on_connect(self, client, _userdata, _flags, rc, _properties=None):
        if _as_int(rc) == 0:
            client.subscribe([(SCENE_TOPIC, 0)])
            self._broadcast({"ok": True, "event": "demonstrator_ready", "rules": len(RULES)})
        else:
            self._broadcast({"ok": False, "event": "connect_failed", "code": _as_int(rc)})

    def on_disconnect(self, _client, _userdata, _disconnect_flags, reason_code, _properties=None):
        self._broadcast({"ok": False, "event": "disconnected", "code": _as_int(reason_code)})

    def on_message(self, _client, _userdata, msg):
        with self.lock:
            if msg.topic != SCENE_TOPIC:
                return
            payload = msg.payload.decode("utf-8", "ignore")
            try:
                scene = json.loads(payload)
            except Exception:
                return
            if not isinstance(scene, dict) or not scene.get("ok"):
                return
            if scene.get("event") != "scene_update":
                return
            action = self._choose_action(scene)
            if action:
                self._emit_action(action, scene)

    # Core logic ------------------------------------------------------
    def _choose_action(self, scene: dict) -> Optional[str]:
        now = time.time()
        if now - self.last_action_ts < ACTION_INTERVAL:
            return None
        if DEMO_REQUIRE_IN_GAME:
            flags = scene.get("flags") or {}
            if flags.get("in_game") is False:
                return None
        text_entries = scene.get("text") or []
        text_blob = " ".join(str(item) for item in text_entries)
        mean = scene.get("mean")
        for rule in RULES:
            if rule.matches(text_blob, mean):
                return rule.action
        # fallback heuristic: alternate move/click to keep dataset varied
        return "click_primary" if self.last_action == "move_cursor" else "move_cursor"

    def _emit_action(self, action: str, scene: dict):
        now = time.time()
        if action == self.last_action:
            self.repeat_count += 1
            if self.repeat_count > MAX_REPEAT:
                return
        else:
            self.repeat_count = 1
        if action == "pause":
            return
        payload = {
            "ok": True,
            "source": AGENT_NAME,
            "action": action,
            "requested_action": action,
            "scene_mean": scene.get("mean"),
            "text": scene.get("text"),
            "timestamp": now,
        }
        act_payload = dict(payload)
        if action == "move_cursor":
            dx = self._random_delta()
            dy = self._random_delta()
            act_payload["action"] = "mouse_move"
            act_payload["dx"] = dx
            act_payload["dy"] = dy
        elif action == "press_enter":
            if DEMO_ALLOW_KEYS:
                act_payload["action"] = "key_press"
                act_payload["key"] = "enter"
            else:
                act_payload["action"] = "click_primary"
        elif action == "focus_field":
            act_payload["action"] = "click_primary"
        elif action == "scroll_down":
            act_payload["action"] = "mouse_move"
            act_payload["dx"] = 0
            act_payload["dy"] = self._random_delta()
        for topic in _act_topics():
            self.client.publish(topic, json.dumps(act_payload), qos=0)
        if CONTROL_TOPIC:
            self.client.publish(CONTROL_TOPIC, json.dumps({"source": AGENT_NAME, "action": action}), qos=0)
        self.last_action = action
        self.last_action_ts = now

    def _random_delta(self) -> int:
        delta = self.rng.randint(-MOUSE_RANGE, MOUSE_RANGE)
        if delta >= 0:
            return max(MIN_MOUSE_DELTA, delta)
        return min(-MIN_MOUSE_DELTA, delta)

    def _broadcast(self, message: dict):
        # status topic keeps things simple
        self.client.publish("demonstrator/status", json.dumps(message), qos=0)

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def _act_topics() -> Iterable[str]:
    topics = {ACT_CMD_TOPIC}
    if ACT_CMD_ALIAS and ACT_CMD_ALIAS != ACT_CMD_TOPIC:
        topics.add(ACT_CMD_ALIAS)
    return topics


def _handle_signal(_signum, _frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    Demonstrator().start()


if __name__ == "__main__":
    main()
