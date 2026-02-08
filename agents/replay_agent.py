#!/usr/bin/env python3
"""Replay buffer agent that stores transitions and serves prioritized samples."""
import json
import logging
import os
import random
import signal
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
STORE_TOPIC = os.getenv("REPLAY_STORE_TOPIC", "replay/store")
SAMPLE_REQ_TOPIC = os.getenv("REPLAY_SAMPLE_TOPIC", "replay/sample-request")
SAMPLE_RESP_TOPIC = os.getenv("REPLAY_SAMPLE_RESP_TOPIC", "replay/sample-response")
BUFFER_SIZE = int(os.getenv("REPLAY_BUFFER_SIZE", "5000"))
MIN_PRIORITY = float(os.getenv("REPLAY_MIN_PRIORITY", "1e-3"))
BUFFER_PATH = Path(os.getenv("REPLAY_BUFFER_PATH", "/mnt/ssd/memory/replay_buffer.json"))
HER_RATIO = float(os.getenv("REPLAY_HER_RATIO", "0.2"))

# --- Setup ---
logging.basicConfig(level=os.getenv("REPLAY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("replay_agent")
stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class ReplayAgent:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="replay_agent")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.priority_sum = 0.0
        self._load_buffer()

    def _load_buffer(self):
        if not BUFFER_PATH.exists(): return
        try:
            with BUFFER_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.buffer.extend(data)
                    self.priority_sum = sum(item.get("priority", 0.0) for item in self.buffer)
                    logger.info("Loaded %s transitions from %s", len(self.buffer), BUFFER_PATH)
        except Exception as e:
            logger.warning("Failed to load replay buffer: %s", e)

    def _save_buffer(self):
        try:
            BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with BUFFER_PATH.open("w", encoding="utf-8") as f:
                json.dump(list(self.buffer), f)
            logger.info("Saved %s transitions to %s", len(self.buffer), BUFFER_PATH)
        except Exception as e:
            logger.error("Failed to save replay buffer: %s", e)

    def _add_transition(self, data):
        priority = max(float(data.get("priority", 1.0)), MIN_PRIORITY)
        transition = {
            "timestamp": data.get("timestamp", time.time()),
            "state": data.get("state"),
            "next_state": data.get("next_state"),
            "action": data.get("action"),
            "reward": data.get("reward"),
            "teacher": data.get("teacher"),
            "task": data.get("task"),
            "extra": data.get("extra"),
            "priority": priority,
        }
        self._push_to_buffer(transition)
        
        # Hindsight Experience Replay (HER)
        # Create a synthetic "success" experience where the achieved state was the goal.
        if random.random() < HER_RATIO:
            her_transition = transition.copy()
            her_transition["her"] = True
            # In a full implementation, we would recompute the reward here assuming
            # the goal was achieved. For now, we flag it for the trainer.
            self._push_to_buffer(her_transition)

    def _push_to_buffer(self, item):
        if len(self.buffer) == BUFFER_SIZE:
            popped = self.buffer.popleft()
            self.priority_sum -= popped.get("priority", 0.0)
        self.buffer.append(item)
        self.priority_sum += item.get("priority", MIN_PRIORITY)

    def _sample_transitions(self, count: int):
        if not self.buffer: return []
        if self.priority_sum <= 0: return random.sample(list(self.buffer), min(count, len(self.buffer)))
        samples = []
        for _ in range(min(count, len(self.buffer))):
            target = random.uniform(0, self.priority_sum)
            cumulative, chosen = 0.0, self.buffer[-1]
            for transition in self.buffer:
                cumulative += transition.get("priority", MIN_PRIORITY)
                if cumulative >= target: chosen = transition; break
            samples.append(chosen)
        return samples

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(STORE_TOPIC, 0), (SAMPLE_REQ_TOPIC, 0)])
            client.publish(SAMPLE_RESP_TOPIC, json.dumps({"ok": True, "event": "replay_ready", "size": len(self.buffer)}))
            logger.info("Replay agent connected")
        else:
            client.publish(SAMPLE_RESP_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))
            logger.error("Connect failed rc=%s", _as_int(rc))

    def _on_message(self, client, userdata, msg):
        try: data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: data = {"raw": msg.payload}
        if msg.topic == STORE_TOPIC: self._add_transition(data)
        elif msg.topic == SAMPLE_REQ_TOPIC:
            samples = self._sample_transitions(int(data.get("count", 32)))
            client.publish(SAMPLE_RESP_TOPIC, json.dumps({"ok": True, "event": "replay_samples", "request_id": data.get("request_id"), "samples": samples, "timestamp": time.time()}))

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self._save_buffer()
        logger.info("Replay agent shut down")

def _handle_signal(signum, frame):
    logger.info("Signal %s received", signum)
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    ReplayAgent().run()

if __name__ == "__main__":
    main()
