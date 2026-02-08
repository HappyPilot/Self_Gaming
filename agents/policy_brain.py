#!/usr/bin/env python3
"""
policy_brain: shadow-mode brain
- Subscribes to scene/state
- Vectorizes state deterministically to fixed size
- Runs a lightweight heuristic model
- Publishes to policy_brain/cmd
- Compares with act/cmd and logs metrics to policy_brain/metrics
"""
from __future__ import annotations

import json
import logging
import os
import random
import signal
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import paho.mqtt.client as mqtt

from utils.state_vectorizer import StateVectorizer

logging.basicConfig(level=os.getenv("POLICY_BRAIN_LOG_LEVEL", "INFO"))
logger = logging.getLogger("policy_brain")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

SCENE_TOPIC = os.getenv("POLICY_BRAIN_SCENE_TOPIC", os.getenv("SCENE_TOPIC", "scene/state"))
ACT_TOPIC = os.getenv("POLICY_BRAIN_ACT_TOPIC", os.getenv("ACT_CMD_TOPIC", "act/cmd"))
PUBLISH_TOPIC = os.getenv("POLICY_BRAIN_PUB_TOPIC", "policy_brain/cmd")
METRIC_TOPIC = os.getenv("POLICY_BRAIN_METRIC_TOPIC", "policy_brain/metrics")

VECTOR_SIZE = int(os.getenv("POLICY_BRAIN_VECTOR_SIZE", "512"))
SCHEMA_VERSION = int(os.getenv("POLICY_BRAIN_SCHEMA_VERSION", "1"))
DEVICE = os.getenv("POLICY_BRAIN_DEVICE", "cpu")
CMD_WINDOW_SEC = float(os.getenv("POLICY_BRAIN_CMD_WINDOW_SEC", "1.0"))
TEXT_TARGET_PROB = float(os.getenv("POLICY_BRAIN_TEXT_TARGET_PROB", "0.5"))
TEXT_TARGET_RNG = random.Random()

MIN_OBJECT_CONF = float(os.getenv("POLICY_BRAIN_MIN_OBJECT_CONF", "0.2"))
IDLE_ACTION = os.getenv("POLICY_BRAIN_IDLE_ACTION", "wait")
ENEMY_LABELS = {
    item.strip().lower()
    for item in os.getenv("POLICY_BRAIN_ENEMY_LABELS", "enemy,boss,monster,bandit").split(",")
    if item.strip()
}
INTERACT_LABELS = {
    item.strip().lower()
    for item in os.getenv("POLICY_BRAIN_INTERACT_LABELS", "portal,waypoint,loot,npc,dialog_button").split(",")
    if item.strip()
}
IGNORE_TEXT_TOKENS = {
    item.strip().lower()
    for item in os.getenv(
        "POLICY_BRAIN_IGNORE_TEXT",
        "quest,objective,mission,completed,menu,gate,mana,life,xp,level,score",
    ).split(",")
    if item.strip()
}

stop_event = False


class HeuristicPolicyModel:
    """Rule-based policy brain to avoid noop-only outputs."""

    def __init__(self):
        self.last_death_click = 0.0

    def __call__(self, scene: Dict, vec: Optional[np.ndarray] = None) -> Dict:
        del vec  # Reserved for future learned models.
        flags = scene.get("flags") or {}
        if flags.get("in_game") is False:
            return _action("wait", 0.1, "not_in_game")
        if flags.get("death"):
            now = time.time()
            if now - self.last_death_click < 2.0:
                return _action("wait", 0.1, "death_cooldown")
            self.last_death_click = now
            # Target common respawn button area (lower middle)
            target = {"label": "respawn_button", "x": 0.5, "y": 0.82}
            return _action("click_primary", 0.6, "death_flag", target)

        enemies = _filter_objects(scene.get("enemies") or [], ENEMY_LABELS)
        if enemies:
            target = _target_from_entry(enemies[0])
            return _action("click_primary", 0.7, "enemy_present", target)

        targets = scene.get("targets") or []
        # Only consider text targets probabilistically to allow for exploration.
        if targets and TEXT_TARGET_PROB > 0 and TEXT_TARGET_RNG.random() <= TEXT_TARGET_PROB:
            target = _select_target(targets)
            if target:
                return _action("mouse_move", 0.5, "text_target", target)

        objects = _filter_objects(scene.get("objects") or [], INTERACT_LABELS)
        if objects:
            target = _target_from_entry(objects[0])
            return _action("mouse_move", 0.35, "object_present", target)

        return _action(IDLE_ACTION, 0.1, "idle")


def _action(action_type: str, confidence: float, reason: str, target: Optional[Dict] = None) -> Dict:
    payload = {"action_type": action_type, "confidence": round(confidence, 3), "reason": reason}
    if target:
        payload["target"] = target
    return payload


def _filter_objects(entries, labels):
    filtered = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        conf = entry.get("confidence")
        try:
            conf_val = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_val = 0.0
        if conf_val < MIN_OBJECT_CONF:
            continue
        label = str(entry.get("label") or entry.get("class") or "").lower()
        if labels and label not in labels:
            continue
        filtered.append(entry)
    return filtered


def _select_target(targets):
    # UI ignore patterns for targeting
    ui_ignore = {"quest", "objective", "mana", "life", "level", "xp", "energy", "gate", "job", "egg"}
    
    for target in targets:
        if not isinstance(target, dict):
            continue
        label_raw = str(target.get("label") or "").strip()
        if not label_raw or len(label_raw) < 2:
            continue
            
        # Filter 1: Too long text is usually a description, not a button
        if len(label_raw) > 25:
            continue
            
        # Filter 2: ALL CAPS text is often a header in games like PoE
        if label_raw.isupper() and len(label_raw) > 4:
            continue
            
        lowered = label_raw.lower()
        compact = "".join(ch for ch in lowered if ch.isalnum())
        
        # Filter 3: Keywords
        if any(token in lowered or token in compact for token in ui_ignore):
            continue
            
        center = target.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            return {
                "label": label_raw,
                "x": float(center[0]),
                "y": float(center[1]),
            }
    return None


def _target_from_entry(entry):
    bbox = entry.get("bbox") or entry.get("box")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            return {"label": entry.get("label"), "x": round((x1 + x2) / 2.0, 4), "y": round((y1 + y2) / 2.0, 4)}
        except (TypeError, ValueError):
            return None
    return None


class PolicyBrainAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="policy_brain", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.vectorizer = StateVectorizer(size=VECTOR_SIZE, schema_version=SCHEMA_VERSION)
        self.model = HeuristicPolicyModel()

        self.scene_buffer: Deque[Dict] = deque(maxlen=4)
        self.last_cmd: Optional[Dict] = None

    # MQTT handlers ------------------------------------------------------
    def _on_connect(self, client, _userdata, _flags, rc):
        topics = [(SCENE_TOPIC, 0), (ACT_TOPIC, 0)]
        for t, qos in topics:
            client.subscribe(t, qos=qos)
        logger.info("Connected to MQTT, subscribed to %s", [t for t, _ in topics])

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == SCENE_TOPIC:
            try:
                payload = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            self.scene_buffer.append(payload)
            self._handle_scene(payload)
        elif msg.topic == ACT_TOPIC:
            try:
                act = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            self._compare_with_act(act)

    # Core logic --------------------------------------------------------
    def _handle_scene(self, scene: Dict):
        vec = self.vectorizer.vectorize(scene)
        cmd = self.model(scene, vec)
        now = time.time()
        payload = {
            "ts": now,
            "source": "policy_brain",
            "action": cmd,
            "vector_schema": SCHEMA_VERSION,
            "vector_size": VECTOR_SIZE,
        }
        self.last_cmd = {"ts": now, "cmd": cmd}
        self.client.publish(PUBLISH_TOPIC, json.dumps(payload))

    def _compare_with_act(self, act: Dict):
        if not self.last_cmd:
            return
        now = time.time()
        if abs(now - self.last_cmd["ts"]) > CMD_WINDOW_SEC:
            return
        act_action = act.get("action") or act.get("policy_action") or act.get("action_type")
        brain_action = self.last_cmd["cmd"].get("action_type")
        match = 1 if act_action == brain_action else 0
        metric = {
            "ts": now,
            "match_action_type": match,
            "act_action": act_action,
            "brain_action": brain_action,
            "source": "policy_brain",
        }
        self.client.publish(METRIC_TOPIC, json.dumps(metric))

    # Run ---------------------------------------------------------------
    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        try:
            while not stop_event:
                time.sleep(0.1)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def _handle_signal(signum, _frame):
    global stop_event
    logger.info("Signal %s received, shutting down.", signum)
    stop_event = True


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    PolicyBrainAgent().run()


if __name__ == "__main__":
    main()
