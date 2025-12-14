#!/usr/bin/env python3
"""Reward manager that fuses PoE logs + scene metrics into dense rewards."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

LOG_PATH = Path(os.getenv("POE_LOG_PATH", "/mnt/ssd/poe_client.txt"))
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
POE_STAGE = os.getenv("POE_STAGE", "S1")
EVENT_WINDOW = int(os.getenv("REWARD_EVENT_WINDOW", "50"))

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
    "currency_high": 6.0,
    "currency_med": 3.5,
    "currency_low": 1.0,
    "rare_item": 1.2,
    "unique_item": 4.0,
    "map": 2.5,
    "fragment": 2.0,
    "essence": 1.0,
    "scarab": 3.0,
    "fossil": 2.5,
    "heist_contract": 2.0,
    "div_card": 1.8,
}

SCENE_OBJECT_VALUES = {
    **LOOT_VALUE_PROFILE,
    "portal": 0.5,
}

ENEMY_LABELS = {"enemy", "elite", "boss", "monster", "rare", "unique_enemy"}

LOG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("area", re.compile(r"you have entered (?P<area>.+)", re.IGNORECASE)),
    ("kill", re.compile(r"you have killed (?P<target>.+)", re.IGNORECASE)),
    ("death", re.compile(r"you have been slain", re.IGNORECASE)),
    ("level", re.compile(r"you have gained a level", re.IGNORECASE)),
    ("quest", re.compile(r"quest complete", re.IGNORECASE)),
    ("loot", re.compile(r"picked up (?P<item>.+)", re.IGNORECASE)),
]


@dataclass
class SceneMetrics:
    """Summary of the latest scene observation."""

    enemy_count: int = 0
    loot_score: float = 0.0
    timestamp: float = 0.0


class PoELogTailer:
    """Tail utility that parses structured events from Path of Exile logs."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.inode = None

    def read_events(self) -> List[dict]:
        if not self.path.exists():
            return []

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
            enemy_class = "boss"
        elif "unique" in label:
            enemy_class = "boss"
        elif any(key in label for key in ("rare", "magic", "metamorph", "harbinger")):
            enemy_class = "elite"
        else:
            enemy_class = "normal"
        return {"target": target, "class": enemy_class}


def classify_item(item_name: str) -> dict:
    """Map loot text to a value bucket defined in the reward spec."""

    name = item_name.lower()
    if any(token in name for token in ("divine orb", "exalted orb", "mirror shard")):
        bucket = "currency_high"
    elif any(token in name for token in ("chaos orb", "ancient orb", "blessed orb", "cartographer's")):
        bucket = "currency_med"
    elif "orb" in name or "scroll" in name:
        bucket = "currency_low"
    elif "scarab" in name:
        bucket = "scarab"
    elif "fossil" in name or "resonator" in name:
        bucket = "fossil"
    elif "essence" in name:
        bucket = "essence"
    elif "map" in name:
        bucket = "map"
    elif "fragment" in name or "splinter" in name:
        bucket = "fragment"
    elif "contract" in name or "blueprint" in name:
        bucket = "heist_contract"
    elif "card" in name:
        bucket = "div_card"
    elif "unique" in name:
        bucket = "unique_item"
    else:
        bucket = "rare_item"
    value = LOOT_VALUE_PROFILE.get(bucket, 0.5)
    return {"bucket": bucket, "value": value, "item": item_name}


def derive_scene_metrics(scene_payload: Optional[dict]) -> SceneMetrics:
    if not isinstance(scene_payload, dict):
        return SceneMetrics()
    objects = scene_payload.get("objects") or []
    enemy_count = 0
    loot_score = 0.0
    for obj in objects:
        label = str(obj.get("class") or "").lower()
        if label in ENEMY_LABELS:
            enemy_count += 1
        if label in SCENE_OBJECT_VALUES:
            loot_score += SCENE_OBJECT_VALUES[label]
    return SceneMetrics(enemy_count=enemy_count, loot_score=loot_score, timestamp=scene_payload.get("timestamp", time.time()))


class RewardCalculator:
    """Implements the dense reward logic described in poe_reward_system_v1.md."""

    def __init__(self, stage: str = "S1"):
        self.stage = stage if stage in STAGE_WEIGHTS else "S1"

    def set_stage(self, stage: str):
        if stage in STAGE_WEIGHTS:
            self.stage = stage

    def compute(self, events: List[dict], metrics: SceneMetrics) -> Tuple[float, Dict[str, float]]:
        stats = {
            "area": 0,
            "quests": 0,
            "levels": 0,
            "normal_kills": 0,
            "elite_kills": 0,
            "boss_kills": 0,
            "loot_value": metrics.loot_score,
            "deaths": 0,
        }

        for evt in events:
            etype = evt.get("type")
            meta = evt.get("meta", {})
            if etype == "area":
                stats["area"] += 1
            elif etype == "quest":
                stats["quests"] += 1
            elif etype == "level":
                stats["levels"] += 1
            elif etype == "kill":
                enemy_class = meta.get("class", "normal")
                key = f"{enemy_class}_kills"
                if key in stats:
                    stats[key] += 1
            elif etype == "loot":
                stats["loot_value"] += float(meta.get("value", 0.0))
            elif etype == "death":
                stats["deaths"] += 1

        weights = STAGE_WEIGHTS[self.stage]

        progress_score = stats["area"] * 0.4 + stats["quests"] * 0.6 + stats["levels"] * 0.2 + stats["boss_kills"] * 0.8
        combat_score = (
            stats["normal_kills"] * 0.02
            + stats["elite_kills"] * 0.08
            + stats["boss_kills"] * 0.25
        )
        loot_score = stats["loot_value"] * 0.05
        enemy_ratio = min(1.0, metrics.enemy_count / ENEMY_CLEAR_TARGET) if ENEMY_CLEAR_TARGET else 0.0
        map_score = max(0.0, 1.0 - enemy_ratio)
        survival_score = -stats["deaths"] * DEATH_PENALTY[self.stage]
        step_penalty = STEP_COST[self.stage]

        total = (
            progress_score * weights["prog"]
            + combat_score * weights["combat"]
            + loot_score * weights["loot"]
            + map_score * weights["map"]
            + survival_score * weights["surv"]
            - step_penalty
        )
        clipped = max(-1.0, min(1.0, total))
        components = {
            "progress": round(progress_score * weights["prog"], 4),
            "combat": round(combat_score * weights["combat"], 4),
            "loot": round(loot_score * weights["loot"], 4),
            "map": round(map_score * weights["map"], 4),
            "survival": round(survival_score * weights["surv"], 4),
            "step_cost": round(-step_penalty, 4),
            "total": round(clipped, 4),
        }
        return clipped, components


class RewardManager:
    """Coordinates MQTT IO, scene buffering, and reward publication."""

    def __init__(self):
        self.client = mqtt.Client(client_id="reward_manager", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.tailer = PoELogTailer(LOG_PATH)
        self.calculator = RewardCalculator(stage=POE_STAGE)
        self.latest_scene: Optional[dict] = None

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(SCENE_TOPIC, 0), (ACT_RESULT_TOPIC, 0)])
            client.publish(REWARD_TOPIC, json.dumps({"ok": True, "event": "reward_manager_ready"}))
        else:
            client.publish(REWARD_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            self.client.publish(REWARD_TOPIC, json.dumps({"ok": False, "event": "disconnected", "code": int(rc)}))

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == SCENE_TOPIC:
            try:
                self.latest_scene = json.loads(msg.payload.decode("utf-8"))
            except json.JSONDecodeError:
                self.latest_scene = None
        self.publish_reward()

    def publish_reward(self):
        events = self.tailer.read_events()
        metrics = derive_scene_metrics(self.latest_scene)
        reward, components = self.calculator.compute(events, metrics)
        payload = {
            "ok": True,
            "reward": round(reward, 4),
            "stage": self.calculator.stage,
            "components": components,
            "events": events[-5:],
            "timestamp": time.time(),
        }
        self.client.publish(REWARD_TOPIC, json.dumps(payload))


def main():
    RewardManager().start()


if __name__ == "__main__":
    main()
