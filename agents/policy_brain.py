#!/usr/bin/env python3
"""Skeleton policy_brain service: aggregates observations and publishes safe actions.

Current behavior: subscribes to vision/observation, ocr_easy/text, scene/state, cursor/state;
keeps a short buffer of events; publishes noop or masked actions to act/cmd. Designed to be
extended with multimodal/SNN policy. Uses control profiles (LLM/onboarding) as action masks.
"""
from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

from control_profile import load_profile, safe_profile

logging.basicConfig(level=os.getenv("POLICY_BRAIN_LOG_LEVEL", "INFO"))
logger = logging.getLogger("policy_brain")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

OBS_TOPIC = os.getenv("VISION_OBS_TOPIC", "vision/observation")
OCR_TOPIC = os.getenv("OCR_TEXT_TOPIC", "ocr_easy/text")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")

GAME_SCHEMA_TOPIC = os.getenv("GAME_SCHEMA_TOPIC", "game/schema")
CONTROL_PROFILE_PATH = os.getenv("CONTROL_PROFILE_PATH", "/app/data/control_profiles.json")
CONTROL_PROFILE_PATH_OBJ = Path(CONTROL_PROFILE_PATH)

ACTION_LOG_PATH = os.getenv("POLICY_BRAIN_LOG_PATH", "/app/logs/policy_brain.log")
ACTION_INTERVAL = float(os.getenv("POLICY_BRAIN_ACTION_INTERVAL", "1.0"))
STATE_TTL_SEC = float(os.getenv("POLICY_BRAIN_STATE_TTL", "3.0"))
TARGET_PRIORITIES = [token.strip().lower() for token in os.getenv("POLICY_BRAIN_TARGETS", "enemy,boss,quest_marker,portal,loot").split(",") if token.strip()]
TARGET_FALLBACK = os.getenv("POLICY_BRAIN_FALLBACK", "mouse_move")
MIN_TARGET_CONF = float(os.getenv("POLICY_BRAIN_MIN_TARGET_CONF", "0.3"))
QUEST_TEXT_HINTS = [t.strip().lower() for t in os.getenv("POLICY_BRAIN_QUEST_HINTS", "quest,objective,target,kill,boss").split(",") if t.strip()]
EXPLORE_JITTER = int(os.getenv("POLICY_BRAIN_EXPLORE_JITTER", "25"))
DEFAULT_PLAY_AREA = {
    "x": float(os.getenv("POLICY_PLAY_X", "0.15")),
    "y": float(os.getenv("POLICY_PLAY_Y", "0.15")),
    "w": float(os.getenv("POLICY_PLAY_W", "0.7")),
    "h": float(os.getenv("POLICY_PLAY_H", "0.7")),
}

stop_event = threading.Event()


def _append_log(line: str):
    if not ACTION_LOG_PATH:
        return
    try:
        os.makedirs(os.path.dirname(ACTION_LOG_PATH), exist_ok=True)
        with open(ACTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


class PolicyBrain:
    def __init__(self):
        self.client = mqtt.Client(client_id="policy_brain", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.profile = safe_profile()
        self.game_id = "unknown_game"
        self.state_buffer: Deque[Dict] = deque(maxlen=200)
        self.ocr_buffer: Deque[Dict] = deque(maxlen=50)
        self.cursor_state: Optional[Dict] = None
        self.last_action_ts = 0.0
        self.last_seen_targets = defaultdict(float)

        self.lock = threading.Lock()

    # MQTT wiring ---------------------------------------------------------
    def _on_connect(self, client, _userdata, _flags, rc):
        topics = [
            (OBS_TOPIC, 0),
            (OCR_TOPIC, 0),
            (SCENE_TOPIC, 0),
            (CURSOR_TOPIC, 0),
            (GAME_SCHEMA_TOPIC, 0),
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        logger.info("Connected, subscribed to %s", [t for t, _ in topics])
        self._publish_ready()

    def _on_message(self, _client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        with self.lock:
            if msg.topic == OBS_TOPIC:
                data["_ts"] = time.time()
                self.state_buffer.append(data)
            elif msg.topic == OCR_TOPIC:
                data_ts = data if isinstance(data, dict) else {"text": str(data)}
                data_ts["_ts"] = time.time()
                self.ocr_buffer.append(data_ts)
            elif msg.topic == CURSOR_TOPIC:
                self.cursor_state = data
            elif msg.topic == SCENE_TOPIC:
                if isinstance(data, dict) and data.get("game_id"):
                    self.game_id = str(data.get("game_id"))
            elif msg.topic == GAME_SCHEMA_TOPIC:
                self._update_profile_from_schema(data)

    # Profile -------------------------------------------------------------
    def _update_profile_from_schema(self, payload: dict):
        schema = payload.get("schema") if isinstance(payload, dict) else None
        if isinstance(schema, dict):
            game_id = str(schema.get("game_id") or self.game_id or "unknown_game")
            profile = schema.get("profile")
            if isinstance(profile, dict):
                self.profile = profile
                self.game_id = game_id
                self._publish_event({"ok": True, "event": "profile_loaded", "game_id": game_id})
                return
        if self.game_id:
            loaded = load_profile(self.game_id, path=CONTROL_PROFILE_PATH_OBJ) if CONTROL_PROFILE_PATH_OBJ else load_profile(self.game_id)
            if loaded:
                self.profile = loaded

    # Publishing ----------------------------------------------------------
    def _publish_ready(self):
        self._publish_event({"ok": True, "event": "policy_brain_ready", "game_id": self.game_id})

    def _publish_event(self, payload: dict):
        try:
            self.client.publish("act/result", json.dumps(payload))
        except Exception:
            return

    def _publish_action(self, action: dict):
        payload = dict(action)
        payload.setdefault("ok", True)
        try:
            self.client.publish(ACT_CMD_TOPIC, json.dumps(payload))
            _append_log(f"{time.time():.3f} action={payload}")
        except Exception:
            logger.exception("Failed to publish action")

    # Core loop -----------------------------------------------------------
    def _build_state_snapshot(self) -> Dict:
        now = time.time()
        with self.lock:
            obs = [s for s in self.state_buffer if now - s.get("_ts", now) <= STATE_TTL_SEC]
            ocr = [o for o in self.ocr_buffer if now - o.get("_ts", now) <= STATE_TTL_SEC]
            cursor = self.cursor_state
            game_id = self.game_id
            profile = self.profile
        return {"obs": obs, "ocr": ocr, "cursor": cursor, "game_id": game_id, "profile": profile}

    def _find_target(self, obs_list: List[dict]) -> Optional[Tuple[float, float, str, float]]:
        """Pick highest-priority non-player object; return (cx, cy, label, conf)."""
        now = time.time()
        best: Optional[Tuple[float, float, str, float]] = None
        best_priority = 999
        for obs in obs_list:
            objects = obs.get("yolo_objects") or []
            for obj in objects:
                label = str(obj.get("label") or obj.get("class") or "").lower()
                if label in {"player", "self", "cursor"}:
                    continue
                try:
                    conf = float(obj.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                if conf < MIN_TARGET_CONF:
                    continue
                priority = TARGET_PRIORITIES.index(label) if label in TARGET_PRIORITIES else len(TARGET_PRIORITIES) + 1
                if priority > best_priority:
                    continue
                bbox = obj.get("bbox") or []
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    cx = max(0.0, min(1.0, float(x) + float(w) / 2.0))
                    cy = max(0.0, min(1.0, float(y) + float(h) / 2.0))
                    best = (cx, cy, label, conf)
                    best_priority = priority
                    self.last_seen_targets[label] = now
        return best

    def _find_quest_text(self, ocr_list: List[dict]) -> Optional[str]:
        for entry in ocr_list:
            text = str(entry.get("text") or "").lower()
            if any(token in text for token in QUEST_TEXT_HINTS):
                return text
        return None

    def _sample_play_area(self, state: Dict) -> Tuple[float, float]:
        layout = None
        with self.lock:
            # Try latest state buffer for ui_layout
            if self.state_buffer:
                layout = self.state_buffer[-1].get("ui_layout")
        if not isinstance(layout, dict):
            layout = DEFAULT_PLAY_AREA
        x = float(layout.get("x", DEFAULT_PLAY_AREA["x"]))
        y = float(layout.get("y", DEFAULT_PLAY_AREA["y"]))
        w = float(layout.get("w", DEFAULT_PLAY_AREA["w"]))
        h = float(layout.get("h", DEFAULT_PLAY_AREA["h"]))
        # clip to [0,1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.05, min(1.0 - x, w))
        h = max(0.05, min(1.0 - y, h))
        rx = random.uniform(x, x + w)
        ry = random.uniform(y, y + h)
        return rx, ry

    def _choose_action(self, state: Dict) -> Dict:
        """Heuristic: click primary at top-priority target if allowed; else safe fallback."""
        profile = state.get("profile") or {}
        target = self._find_target(state.get("obs") or [])
        if target and profile.get("allow_primary"):
            cx, cy, label, conf = target
            return {"action": "click_primary", "x": cx, "y": cy, "label": label, "conf": conf, "source": "policy_brain"}
        # If no target, explore cautiously: random click within central band or small jitter move.
        if profile.get("allow_primary"):
            cx, cy = self._sample_play_area(state)
            return {"action": "click_primary", "x": cx, "y": cy, "label": "explore_scan", "source": "policy_brain"}
        if profile.get("allow_mouse_move"):
            jitter = max(1, EXPLORE_JITTER)
            dx = random.randint(-jitter, jitter)
            dy = random.randint(-jitter, jitter)
            return {"action": "mouse_move", "dx": dx, "dy": dy, "source": "policy_brain"}
        quest_hint = self._find_quest_text(state.get("ocr") or [])
        return {"action": "noop", "source": "policy_brain", "quest_hint": quest_hint}

    def _loop(self):
        while not stop_event.is_set():
            now = time.time()
            if now - self.last_action_ts >= ACTION_INTERVAL:
                state = self._build_state_snapshot()
                action = self._choose_action(state)
                self._publish_action(action)
                self.last_action_ts = now
            time.sleep(0.05)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        try:
            self._loop()
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def _handle_signal(_signum, _frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    brain = PolicyBrain()
    brain.run()


if __name__ == "__main__":
    main()
