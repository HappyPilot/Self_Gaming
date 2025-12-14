#!/usr/bin/env python3
"""Lightweight top-down simulator service publishing MQTT state updates."""
from __future__ import annotations

import json
import math
import os
import random
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import paho.mqtt.client as mqtt
import logging

# --- Setup ---
logging.basicConfig(level=os.getenv("SIM_CORE_LOG_LEVEL", "INFO"))
logger = logging.getLogger("sim_core")

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
STATE_TOPIC = os.getenv("SIM_STATE_TOPIC", "sim_core/state")
ACTION_TOPIC = os.getenv("SIM_ACTION_TOPIC", "sim_core/action")
CMD_TOPIC = os.getenv("SIM_CMD_TOPIC", "sim_core/cmd")
REWARD_TOPIC = os.getenv("SIM_REWARD_TOPIC", "train/reward")
TICK_SECONDS = float(os.getenv("SIM_TICK_SECONDS", "0.2"))
DEFAULT_DR_LEVEL = os.getenv("SIM_DR_LEVEL", "medium")
MAP_SIZE = float(os.getenv("SIM_MAP_SIZE", "1.0"))

DR_PROFILES = {
    "low": {"enemy_speed_range": (0.01, 0.02), "enemy_spawn_density": 0.05, "map_layout_seed": 1, "loot_drop_rate": 0.1, "visual_noise_level": 0.01, "input_latency_ms": 50},
    "medium": {"enemy_speed_range": (0.02, 0.04), "enemy_spawn_density": 0.1, "map_layout_seed": 42, "loot_drop_rate": 0.2, "visual_noise_level": 0.03, "input_latency_ms": 120},
    "high": {"enemy_speed_range": (0.03, 0.06), "enemy_spawn_density": 0.2, "map_layout_seed": 1337, "loot_drop_rate": 0.3, "visual_noise_level": 0.05, "input_latency_ms": 200},
}

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

@dataclass
class Entity:
    id: str; kind: str; x: float; y: float; hp: float; speed: float = 0.0
    def to_object(self) -> Dict:
        return {"id": self.id, "class": self.kind, "bbox": [self.x - 0.02, self.y - 0.02, 0.04, 0.04], "pos": [self.x, self.y], "hp": round(self.hp, 3)}

class SimCoreAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="sim_core", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.lock = threading.Lock()
        self.dr_level = DEFAULT_DR_LEVEL
        self.params = DR_PROFILES.get(self.dr_level, DR_PROFILES["medium"])
        self.latency_queue: Deque[Dict] = deque()
        self.player = Entity("player", "player", MAP_SIZE / 2, MAP_SIZE / 2, hp=1.0)
        self.enemies: List[Entity] = []
        self.loot: List[Entity] = []
        self.last_reward = 0.0
        self.pending_goal = None
        self.random = random.Random(self.params["map_layout_seed"])
        self.reset_env()

    def on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(ACTION_TOPIC, 0), (CMD_TOPIC, 0)])
            client.publish(STATE_TOPIC, json.dumps({"ok": True, "event": "sim_core_ready", "dr_level": self.dr_level}))
            logger.info("Sim core ready")
        else:
            client.publish(STATE_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))
            logger.error("Connect failed rc=%s", _as_int(rc))

    def on_message(self, client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: return
        with self.lock:
            if msg.topic == ACTION_TOPIC: self.enqueue_action(data)
            elif msg.topic == CMD_TOPIC: self.handle_cmd(data)

    def set_dr_level(self, level: str):
        if level not in DR_PROFILES: return
        self.dr_level = level
        self.params = DR_PROFILES[level]
        self.random.seed(self.params["map_layout_seed"])

    def reset_env(self):
        self.player = Entity("player", "player", MAP_SIZE / 2, MAP_SIZE / 2, hp=1.0)
        self.enemies, self.loot, self.last_reward = [], [], 0.0
        self.latency_queue.clear()
        self.pending_goal = None

    def enqueue_action(self, action: Dict):
        latency = self.params["input_latency_ms"] / 1000.0
        self.latency_queue.append({"action": action, "when": time.time() + latency})

    def handle_cmd(self, data: Dict):
        cmd = (data.get("cmd") or "").lower()
        if cmd == "reset": self.reset_env()
        elif cmd == "set_dr": self.set_dr_level(data.get("level", "medium"))
        elif cmd == "set_goal": self.pending_goal = data.get("goal_id")

    def apply_actions(self):
        now = time.time()
        while self.latency_queue and self.latency_queue[0]["when"] <= now:
            action = self.latency_queue.popleft()["action"]
            label = (action.get("action") or action.get("label") or "").lower()
            if label in {"move", "mouse_move"}: self.move_player(float(action.get("dx", 0.0)) * 0.01, float(action.get("dy", 0.0)) * 0.01)
            elif label in {"attack", "click_primary"}: self.attack_nearest()
            elif label in {"loot", "click_secondary"}: self.pickup_loot()

    def move_player(self, dx: float, dy: float):
        self.player.x = max(0.0, min(MAP_SIZE, self.player.x + dx * 0.05))
        self.player.y = max(0.0, min(MAP_SIZE, self.player.y + dy * 0.05))

    def attack_nearest(self):
        if not self.enemies: return
        enemy = min(self.enemies, key=lambda e: (e.x - self.player.x) ** 2 + (e.y - self.player.y) ** 2)
        enemy.hp -= 0.2
        if enemy.hp <= 0:
            self.last_reward += 1.0; self.enemies.remove(enemy)
            if self.random.random() < self.params["loot_drop_rate"]: self.spawn_loot(enemy.x, enemy.y)

    def pickup_loot(self):
        for loot in list(self.loot):
            if self.distance(self.player, loot) < 0.05:
                self.last_reward += 0.2; self.loot.remove(loot)

    def spawn_enemy(self):
        speed = self.random.uniform(*self.params["enemy_speed_range"])
        self.enemies.append(Entity(f"enemy_{uuid.uuid4().hex[:6]}", "enemy_melee" if self.random.random() < 0.7 else "enemy_ranged",
                                  self.random.uniform(0.0, MAP_SIZE), self.random.uniform(0.0, MAP_SIZE), hp=1.0, speed=speed))

    def spawn_loot(self, x: float, y: float):
        self.loot.append(Entity(f"loot_{uuid.uuid4().hex[:6]}", "loot_currency" if self.random.random() < 0.5 else "loot_rare", x, y, hp=0.0))

    def distance(self, a: Entity, b: Entity) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def update_enemies(self):
        for enemy in list(self.enemies):
            dx, dy = self.player.x - enemy.x, self.player.y - enemy.y
            dist = max(math.sqrt(dx * dx + dy * dy), 1e-6)
            enemy.x += enemy.speed * (dx / dist); enemy.y += enemy.speed * (dy / dist)
            enemy.x, enemy.y = max(0.0, min(MAP_SIZE, enemy.x)), max(0.0, min(MAP_SIZE, enemy.y))
            if self.distance(enemy, self.player) < 0.03:
                self.player.hp = max(0.0, self.player.hp - 0.05)
                if enemy.kind == "enemy_ranged": self.last_reward -= 0.05

    def maybe_spawn_entities(self):
        if self.random.random() < self.params["enemy_spawn_density"]: self.spawn_enemy()
        if self.random.random() < self.params["loot_drop_rate"] * 0.05: self.spawn_loot(self.random.uniform(0.0, MAP_SIZE), self.random.uniform(0.0, MAP_SIZE))

    def build_state(self) -> Dict:
        objects = [e.to_object() for e in self.enemies] + [l.to_object() for l in self.loot]
        stats = {"hp_pct": round(self.player.hp, 3), "enemy_count": len(self.enemies), "loot_count": len(self.loot)}
        mean_luma = 0.5 + self.params["visual_noise_level"] * self.random.uniform(-1, 1)
        return {"ok": True, "mode": "sim", "dr_level": self.dr_level, "stats": stats, "objects": objects,
                "text": [f"Enemies: {len(self.enemies)}", f"Loot: {len(self.loot)}"],
                "mean": round(mean_luma, 3), "goal_id": self.pending_goal, "frame": None}

    def publish_state(self): self.client.publish(STATE_TOPIC, json.dumps(self.build_state()))
    def publish_reward(self):
        if self.last_reward == 0.0: return
        self.client.publish(REWARD_TOPIC, json.dumps({"ok": True, "source": "sim_core", "reward": self.last_reward, "timestamp": time.time()}))
        self.last_reward = 0.0

    def step(self):
        with self.lock:
            self.apply_actions()
            self.update_enemies()
            self.maybe_spawn_entities()
            state = self.build_state()
            reward = self.last_reward
            if self.player.hp <= 0.0: self.reset_env()
        self.client.publish(STATE_TOPIC, json.dumps(state))
        if reward != 0.0:
            self.client.publish(REWARD_TOPIC, json.dumps({"ok": True, "source": "sim_core", "reward": reward, "timestamp": time.time()}))
            with self.lock: self.last_reward = 0.0

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        logger.info("Sim core agent started")
        try:
            while not stop_event.is_set():
                start = time.time()
                self.step()
                elapsed = time.time() - start
                stop_event.wait(max(0.0, TICK_SECONDS - elapsed))
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Sim core agent shut down")

def _handle_signal(signum, frame):
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = SimCoreAgent()
    agent.run()

if __name__ == "__main__":
    main()
