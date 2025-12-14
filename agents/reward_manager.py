#!/usr/bin/env python3
"""Reward manager that fuses PoE logs + scene metrics into dense rewards."""
from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt
import logging

logging.basicConfig(level=os.getenv("REWARD_LOG_LEVEL", "INFO"))
logger = logging.getLogger("reward_manager")

# --- Constants ---
LOG_PATH = Path(os.getenv("POE_LOG_PATH", "/mnt/ssd/poe_client.txt"))
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
PRED_ERROR_TOPIC = os.getenv("PRED_ERROR_TOPIC", "world_model/pred_error")
LEARNING_STAGE = int(os.getenv("LEARNING_STAGE", "1"))
DEFAULT_STAGE = os.getenv("POE_STAGE", "S1")
POE_STAGE = "S0" if LEARNING_STAGE == 0 else DEFAULT_STAGE
EVENT_WINDOW = int(os.getenv("REWARD_EVENT_WINDOW", "50"))
DEATH_STATE_PENALTY = float(os.getenv("DEATH_STATE_PENALTY", "0.05"))
DEATH_RECOVERY_BONUS = float(os.getenv("DEATH_RECOVERY_BONUS", "0.4"))
DEATH_SCOPE = os.getenv("REWARD_DEATH_SCOPE", "critical_dialog:death")
CURIOSITY_WEIGHT = float(os.getenv("CURIOSITY_WEIGHT", "0.1"))

# ... (rest of constants and imports unchanged) ...

# (STAGE_WEIGHTS, STEP_COST, etc. unchanged)
STAGE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "S0": {"prog": 0.4, "combat": 0.25, "loot": 0.15, "map": 0.1, "surv": 0.1},
    "S1": {"prog": 0.35, "combat": 0.3, "loot": 0.15, "map": 0.1, "surv": 0.1},
    "S2": {"prog": 0.3, "combat": 0.35, "loot": 0.15, "map": 0.12, "surv": 0.08},
    "S3": {"prog": 0.25, "combat": 0.35, "loot": 0.1, "map": 0.15, "surv": 0.15},
    "S4": {"prog": 0.2, "combat": 0.35, "loot": 0.1, "map": 0.15, "surv": 0.2},
}

STEP_COST = {"S0": 0.002, "S1": 0.0015, "S2": 0.0012, "S3": 0.0010, "S4": 0.0008}
DEATH_PENALTY = {"S0": 0.8, "S1": 1.0, "S2": 1.3, "S3": 1.6, "S4": 2.2}
ENEMY_CLEAR_TARGET = 35

LOOT_VALUE_PROFILE = {
    "currency_high": 6.0, "currency_med": 3.5, "currency_low": 1.0, "rare_item": 1.2,
    "unique_item": 4.0, "map": 2.5, "fragment": 2.0, "essence": 1.0, "scarab": 3.0,
    "fossil": 2.5, "heist_contract": 2.0, "div_card": 1.8,
}

SCENE_OBJECT_VALUES = {**LOOT_VALUE_PROFILE, "portal": 0.5}
ENEMY_LABELS = {"enemy", "elite", "boss", "monster", "rare", "unique_enemy"}

LOG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("area", re.compile(r"you have entered (?P<area>.+)", re.IGNORECASE)),
    ("kill", re.compile(r"you have killed (?P<target>.+)", re.IGNORECASE)),
    ("death", re.compile(r"you have been slain", re.IGNORECASE)),
    ("level", re.compile(r"you have gained a level", re.IGNORECASE)),
    ("quest", re.compile(r"quest complete", re.IGNORECASE)),
    ("loot", re.compile(r"picked up (?P<item>.+)", re.IGNORECASE)),
]

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

@dataclass
class SceneMetrics:
    enemy_count: int = 0
    loot_score: float = 0.0
    timestamp: float = 0.0

# ... (PoELogTailer unchanged) ...
class PoELogTailer:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.inode = None

    def read_events(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            stat = self.path.stat()
            if self.inode != stat.st_ino:
                self.inode = stat.st_ino
                self.offset = 0
            if self.offset > stat.st_size:
                self.offset = 0
            with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(self.offset)
                lines = handle.readlines()
                self.offset = handle.tell()
        except (FileNotFoundError, IOError, OSError) as e:
            logger.warning("Failed to read log file %s: %s", self.path, e)
            return []
        events: List[dict] = []
        for raw_line in lines[-EVENT_WINDOW:]:
            payload = raw_line.split("]", 1)
            message = payload[-1].strip() if payload else raw_line.strip()
            if not message:
                continue
            event = self._parse_line(message)
            if event:
                events.append(event)
        return events

    def _parse_line(self, message: str) -> Optional[dict]:
        lower = message.lower()
        for event_type, pattern in LOG_PATTERNS:
            match = pattern.search(lower)
            if not match:
                continue
            meta = {k: v for k, v in match.groupdict().items() if v}
            if event_type == "kill" and "target" in meta:
                meta.update(self._classify_kill(meta["target"]))
            elif event_type == "loot" and "item" in meta:
                meta.update(classify_item(meta["item"]))
            return {"type": event_type, "meta": meta, "raw": message}
        return None

    @staticmethod
    def _classify_kill(target: str) -> dict:
        label = target.lower()
        if any(key in label for key in ("sirus", "maven", "kitava", "elder", "shaper", "boss")):
            return {"target": target, "class": "boss"}
        if "unique" in label:
            return {"target": target, "class": "boss"}
        if any(key in label for key in ("rare", "magic", "metamorph", "harbinger")):
            return {"target": target, "class": "elite"}
        return {"target": target, "class": "normal"}

# ... (classify_item and derive_scene_metrics unchanged) ...
def classify_item(item_name: str) -> dict:
    name = item_name.lower()
    if any(token in name for token in ("divine orb", "exalted orb", "mirror shard")):
        bucket = "currency_high"
    elif any(token in name for token in ("chaos orb", "ancient orb", "blessed orb", "cartographer's")):
        bucket = "currency_med"
    elif "orb" in name or "scroll" in name:
        bucket = "currency_low"
    elif any(token in name for token in ("scarab", "fossil", "resonator", "essence", "map", "fragment", "splinter", "contract", "blueprint", "card")):
        bucket = token
    else:
        bucket = "rare_item"
    value = LOOT_VALUE_PROFILE.get(bucket, 0.5)
    return {"bucket": bucket, "value": value, "item": item_name}


def derive_scene_metrics(scene_payload: Optional[dict]) -> SceneMetrics:
    if not isinstance(scene_payload, dict):
        return SceneMetrics()
    objects = scene_payload.get("objects") or []
    enemy_count = sum(1 for obj in objects if str(obj.get("class") or "").lower() in ENEMY_LABELS)
    loot_score = sum(SCENE_OBJECT_VALUES.get(str(obj.get("class") or "").lower(), 0.0) for obj in objects)
    return SceneMetrics(enemy_count=enemy_count, loot_score=loot_score, timestamp=scene_payload.get("timestamp", time.time()))

# ... (RewardCalculator unchanged) ...
class RewardCalculator:
    def __init__(self, stage: str = "S1"):
        self.stage = stage if stage in STAGE_WEIGHTS else "S1"

    def set_stage(self, stage: str):
        if stage in STAGE_WEIGHTS:
            self.stage = stage

    def compute(self, events: List[dict], metrics: SceneMetrics) -> Tuple[float, Dict[str, float]]:
        stats = {"area": 0, "quests": 0, "levels": 0, "normal_kills": 0, "elite_kills": 0, "boss_kills": 0, "loot_value": metrics.loot_score, "deaths": 0}
        for evt in events:
            etype, meta = evt.get("type"), evt.get("meta", {})
            if etype == "area": stats["area"] += 1
            elif etype == "quest": stats["quests"] += 1
            elif etype == "level": stats["levels"] += 1
            elif etype == "kill": stats[f"{meta.get('class', 'normal')}_kills"] += 1
            elif etype == "loot": stats["loot_value"] += float(meta.get("value", 0.0))
            elif etype == "death": stats["deaths"] += 1
        weights = STAGE_WEIGHTS[self.stage]
        progress_score = stats["area"] * 0.4 + stats["quests"] * 0.6 + stats["levels"] * 0.2 + stats["boss_kills"] * 0.8
        combat_score = stats["normal_kills"] * 0.02 + stats["elite_kills"] * 0.08 + stats["boss_kills"] * 0.25
        loot_score = stats["loot_value"] * 0.05
        enemy_ratio = min(1.0, metrics.enemy_count / ENEMY_CLEAR_TARGET) if ENEMY_CLEAR_TARGET else 0.0
        map_score = max(0.0, 1.0 - enemy_ratio)
        survival_score = -stats["deaths"] * DEATH_PENALTY[self.stage]
        step_penalty = STEP_COST[self.stage]
        total = (progress_score * weights["prog"] + combat_score * weights["combat"] + loot_score * weights["loot"] +
                 map_score * weights["map"] + survival_score * weights["surv"] - step_penalty)
        clipped = max(-1.0, min(1.0, total))
        return clipped, {"progress": round(progress_score * weights["prog"], 4), "combat": round(combat_score * weights["combat"], 4),
                         "loot": round(loot_score * weights["loot"], 4), "map": round(map_score * weights["map"], 4),
                         "survival": round(survival_score * weights["surv"], 4), "step_cost": round(-step_penalty, 4), "total": round(clipped, 4)}


class RewardManager:
    def __init__(self):
        self.client = mqtt.Client(client_id="reward_manager", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.tailer = PoELogTailer(LOG_PATH)
        self.calculator = RewardCalculator(stage=POE_STAGE)
        self.latest_scene: Optional[dict] = None
        self.last_pred_error = 0.0
        self.death_active = False
        self.death_entered = 0.0

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(SCENE_TOPIC, 0), (ACT_RESULT_TOPIC, 0)])
            if PRED_ERROR_TOPIC:
                client.subscribe([(PRED_ERROR_TOPIC, 0)])
            client.publish(REWARD_TOPIC, json.dumps({"ok": True, "event": "reward_manager_ready"}))
        else:
            logger.error("Reward manager failed to connect: rc=%s", _as_int(rc))
            client.publish(REWARD_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0:
            logger.warning("Reward manager disconnected: rc=%s", _as_int(rc))

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == SCENE_TOPIC:
            try:
                self.latest_scene = json.loads(msg.payload.decode("utf-8"))
            except json.JSONDecodeError:
                self.latest_scene = None
        elif msg.topic == PRED_ERROR_TOPIC:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                if isinstance(data, dict) and "error" in data:
                    self.last_pred_error = float(data["error"])
            except Exception:
                pass
        self.publish_reward()

    def publish_reward(self):
        events = self.tailer.read_events()
        metrics = derive_scene_metrics(self.latest_scene)
        reward, components = self.calculator.compute(events, metrics)
        
        # Intrinsic Curiosity
        intrinsic_reward = self.last_pred_error * CURIOSITY_WEIGHT
        reward += intrinsic_reward
        components["curiosity"] = round(intrinsic_reward, 4)
        
        reward, components = self._apply_stage0_adjustments(reward, components)
        
        # Clip final total
        reward = max(-1.0, min(1.0, reward))
        components["total"] = round(reward, 4)

        payload = {"ok": True, "reward": round(reward, 4), "stage": self.calculator.stage, "components": components,
                   "events": events[-5:], "timestamp": time.time()}
        self.client.publish(REWARD_TOPIC, json.dumps(payload))
        if abs(payload["reward"]) >= 0.01 or events:
            logger.info("Reward event stage=%s reward=%.3f last_event=%s components=%s", self.calculator.stage,
                        payload["reward"], events[-1] if events else None, components)

    def _apply_stage0_adjustments(self, reward: float, components: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        if LEARNING_STAGE != 0:
            return reward, components
        flags = (self.latest_scene or {}).get("flags") or {}
        now = time.time()
        if flags.get("death"):
            if not self.death_active:
                self.death_active = True
                self.death_entered = now
            penalty = min(1.0, (now - self.death_entered) * DEATH_STATE_PENALTY)
            reward -= penalty
            components["death_penalty"] = round(-penalty, 4)
        elif self.death_active:
            reward += DEATH_RECOVERY_BONUS
            components["death_recovery"] = round(DEATH_RECOVERY_BONUS, 4)
            self.death_active = False
            self.death_entered = 0.0
            logger.info("Reward: death->resurrected transition scope=%s bonus=%.3f", DEATH_SCOPE, DEATH_RECOVERY_BONUS)
        return reward, components


def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down reward manager.", signum)
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    RewardManager().start()


if __name__ == "__main__":
    main()
