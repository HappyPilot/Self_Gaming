#!/usr/bin/env python3
"""Aggregate training progress metrics and publish a global status."""
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
TRAIN_STATUS_TOPIC = os.getenv("TRAIN_STATUS_TOPIC", "train/status")
THERMAL_TOPIC = os.getenv("THERMAL_TOPIC", "system/thermal")
PROGRESS_TOPIC = os.getenv("PROGRESS_TOPIC", "progress/status")
RECORDER_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
PINNED_PATH = Path(os.getenv("PINNED_PATH", "/mnt/ssd/memory/pinned.json"))
UPDATE_INTERVAL = int(os.getenv("PROGRESS_UPDATE_INTERVAL", "60"))
REWARD_WINDOW = int(os.getenv("PROGRESS_REWARD_WINDOW", "300"))
DATASET_SCAN_INTERVAL = int(os.getenv("PROGRESS_DATASET_INTERVAL", "180"))
REWARD_TARGET = float(os.getenv("PROGRESS_REWARD_TARGET", "10.0"))
DATASET_TARGET = int(os.getenv("PROGRESS_DATASET_TARGET", "1000"))
PINNED_TARGET = int(os.getenv("PROGRESS_PINNED_TARGET", "25"))
JOB_FRESH_HALF_LIFE = int(os.getenv("PROGRESS_JOB_HALFLIFE", "3600"))  # seconds
LOG_PATH = Path(os.getenv("PROGRESS_LOG_PATH", "/mnt/ssd/logs/progress.log"))

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


class ProgressAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="progress_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.reward_window: Deque[Tuple[float, float]] = deque()
        self.last_job: Dict[str, object] = {}
        self.last_checkpoint_time: Optional[float] = None
        self.thermal_state: str = "unknown"
        self.dataset_count: int = 0
        self.pinned_count: int = 0
        self._last_dataset_scan: float = 0.0
        self._lock = threading.Lock()

    # MQTT callbacks -----------------------------------------------------
    def on_connect(self, client, _userdata, _flags, rc):
        topics = []
        if REWARD_TOPIC:
            topics.append((REWARD_TOPIC, 0))
        if TRAIN_STATUS_TOPIC:
            topics.append((TRAIN_STATUS_TOPIC, 0))
        if THERMAL_TOPIC:
            topics.append((THERMAL_TOPIC, 0))
        for topic in topics:
            client.subscribe(topic)
        status = {"ok": rc == 0, "event": "progress_ready" if rc == 0 else "connect_failed"}
        client.publish(PROGRESS_TOPIC, json.dumps(status))

    def on_message(self, _client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}
        if msg.topic == REWARD_TOPIC:
            self._handle_reward(data)
        elif msg.topic == TRAIN_STATUS_TOPIC:
            self._handle_train_status(data)
        elif msg.topic == THERMAL_TOPIC:
            self.thermal_state = data.get("state", "unknown")

    # Handlers -----------------------------------------------------------
    def _handle_reward(self, data: Dict[str, object]):
        value = data.get("reward") if isinstance(data, dict) else None
        try:
            reward_value = float(value)
        except (TypeError, ValueError):
            reward_value = 0.0
        now = time.time()
        with self._lock:
            self.reward_window.append((now, reward_value))
            self._trim_rewards(now)

    def _handle_train_status(self, data: Dict[str, object]):
        event = data.get("event")
        now = time.time()
        with self._lock:
            self.last_job = {
                "event": event,
                "job_id": data.get("job_id"),
                "samples": data.get("samples"),
                "timestamp": data.get("timestamp") or now,
                "ok": data.get("ok"),
            }
            if event == "job_finished" and data.get("ok"):
                self.last_checkpoint_time = now

    def _trim_rewards(self, now: float):
        while self.reward_window and now - self.reward_window[0][0] > REWARD_WINDOW:
            self.reward_window.popleft()

    # Metrics ------------------------------------------------------------
    def _scan_dataset(self):
        now = time.time()
        if now - self._last_dataset_scan < DATASET_SCAN_INTERVAL:
            return
        try:
            count = sum(1 for _ in RECORDER_DIR.glob("sample_*.json"))
            self.dataset_count = count
        except Exception:
            pass
        try:
            if PINNED_PATH.exists():
                data = json.loads(PINNED_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.pinned_count = len(data)
        except Exception:
            pass
        self._last_dataset_scan = now

    def build_summary(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            self._trim_rewards(now)
            rewards = [val for _ts, val in self.reward_window]
            reward_sum = sum(rewards)
            reward_avg = reward_sum / len(rewards) if rewards else 0.0
            reward_norm = clamp(reward_avg / max(REWARD_TARGET, 1e-6))
            dataset_norm = clamp(self.dataset_count / max(DATASET_TARGET, 1))
            pinned_norm = clamp(self.pinned_count / max(PINNED_TARGET, 1))
            if self.last_job:
                age = now - float(self.last_job.get("timestamp", now))
                job_factor = math.exp(-age / max(JOB_FRESH_HALF_LIFE, 1.0))
            else:
                job_factor = 0.0
            thermal_penalty = {
                "ok": 1.0,
                "warm": 0.85,
                "hot": 0.5,
                "cooldown": 0.6,
            }.get(self.thermal_state, 1.0)
            score = (
                reward_norm * 0.45
                + dataset_norm * 0.2
                + pinned_norm * 0.1
                + job_factor * 0.25
            )
            score *= thermal_penalty
            summary = {
                "ok": True,
                "timestamp": now,
                "reward_sum_window": reward_sum,
                "reward_avg_window": reward_avg,
                "reward_samples_window": len(rewards),
                "reward_window_sec": REWARD_WINDOW,
                "dataset_samples": self.dataset_count,
                "pinned_episodes": self.pinned_count,
                "last_job": self.last_job,
                "thermal_state": self.thermal_state,
                "score": round(score, 3),
            }
        return summary

    # Main loop ----------------------------------------------------------
    def publish_loop(self):
        while True:
            self._scan_dataset()
            summary = self.build_summary()
            payload = json.dumps(summary)
            self.client.publish(PROGRESS_TOPIC, payload)
            try:
                with LOG_PATH.open("a", encoding="utf-8") as logf:
                    logf.write(payload + "\n")
            except Exception:
                pass
            time.sleep(UPDATE_INTERVAL)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        publisher = threading.Thread(target=self.publish_loop, daemon=True)
        publisher.start()
        self.client.loop_forever()


def main():
    agent = ProgressAgent()
    agent.run()


if __name__ == "__main__":
    main()
