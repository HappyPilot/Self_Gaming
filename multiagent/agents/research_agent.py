#!/usr/bin/env python3
"""Research agent that surfaces cached insights and triggers sim experiments."""
from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import List

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
RESEARCH_TOPIC = os.getenv("RESEARCH_TOPIC", "research/events")
GOAL_TOPIC = os.getenv("GOALS_TOPIC", "goals/high_level")
TRAIN_JOB_TOPIC = os.getenv("TRAIN_JOB_TOPIC", "train/jobs")
DOC_DIR = Path(os.getenv("RESEARCH_DOC_DIR", "/mnt/ssd/research/cache"))
INTERVAL = float(os.getenv("RESEARCH_INTERVAL", "120"))
SIM_CMD_TOPIC = os.getenv("SIM_CMD_TOPIC", "sim_core/cmd")
DEFAULT_MODE = os.getenv("RESEARCH_TRAIN_MODE", "ppo_baseline")
VISION_CONFIG_TOPIC = os.getenv("VISION_CONFIG_TOPIC", "vision/config")
VISION_MODES = [m.strip().lower() for m in os.getenv("RESEARCH_VISION_MODES", "low,medium,high").split(",") if m.strip()]


class ResearchAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="research_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.docs: List[str] = []
        self._load_docs()
        self._stop = threading.Event()

    def _load_docs(self):
        if DOC_DIR.exists():
            for path in DOC_DIR.glob("**/*.txt"):
                try:
                    self.docs.append(path.read_text(encoding="utf-8"))
                except Exception:
                    continue

    def on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "research_ready"}))
            threading.Thread(target=self._loop, daemon=True).start()
        else:
            client.publish(RESEARCH_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(INTERVAL)
            self.publish_goal()
            self.launch_sim_experiment()

    def publish_goal(self):
        idea = random.choice(self.docs) if self.docs else "Explore new map layout"
        goal = {
            "ok": True,
            "goal_id": f"research_{int(time.time())}",
            "goal_type": "research",
            "reasoning": idea[:256],
            "source": "research_agent",
            "timestamp": time.time(),
        }
        self.client.publish(GOAL_TOPIC, json.dumps(goal))
        self.client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "goal_proposed", "goal_id": goal["goal_id"]}))

    def launch_sim_experiment(self):
        job_id = f"research_sim_{int(time.time())}"
        mode = DEFAULT_MODE
        target = None
        if random.random() < 0.2:
            mode = "world_model_experiment"
            target = "world_model"
        payload = {
            "ok": True,
            "job_id": job_id,
            "mode": mode,
            "dataset": "sim_core",
        }
        if target:
            payload["target"] = target
        self.client.publish(TRAIN_JOB_TOPIC, json.dumps(payload))
        dr_level = random.choice(["low", "medium", "high"])
        if SIM_CMD_TOPIC:
            self.client.publish(SIM_CMD_TOPIC, json.dumps({"cmd": "set_dr", "level": dr_level}))
        if VISION_CONFIG_TOPIC and VISION_MODES:
            mode_choice = dr_level if dr_level in VISION_MODES else random.choice(VISION_MODES)
            cfg = {
                "mode": mode_choice,
                "reason": "research_experiment",
                "ttl_sec": 15,
            }
            self.client.publish(VISION_CONFIG_TOPIC, json.dumps(cfg))
        self.client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "experiment_started", "job_id": job_id}))

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()


def main():
    agent = ResearchAgent()
    agent.run()


if __name__ == "__main__":
    main()
