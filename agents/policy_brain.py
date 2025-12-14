#!/usr/bin/env python3
"""
policy_brain: shadow-mode brain
- Subscribes to scene/state
- Vectorizes state deterministically to fixed size
- Runs dummy model (noop)
- Publishes to policy_brain/cmd
- Compares with act/cmd and logs metrics to policy_brain/metrics
"""
from __future__ import annotations

import json
import logging
import os
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

stop_event = False


class DummyPolicyModel:
    """Placeholder model: always emits noop."""

    def __call__(self, vec: np.ndarray) -> Dict:
        return {
            "action_type": "noop",
            "confidence": 0.0,
            "reason": "dummy_model",
        }


class PolicyBrainAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="policy_brain", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.vectorizer = StateVectorizer(size=VECTOR_SIZE, schema_version=SCHEMA_VERSION)
        self.model = DummyPolicyModel()

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
        cmd = self.model(vec)
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
