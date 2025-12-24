#!/usr/bin/env python3
"""Reflex agent that consumes observations and strategy state."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time

import json
import paho.mqtt.client as mqtt

from shared_state.strategy_state import build_strategy_state_adapter
from reflex_policy import ReflexPolicyAdapter

logging.basicConfig(level=os.getenv("REFLEX_LOG_LEVEL", "INFO"))
logger = logging.getLogger("reflex_agent")
stop_event = threading.Event()

POLL_SEC = float(os.getenv("REFLEX_POLL_SEC", "0.1"))
LOG_EVERY_SEC = float(os.getenv("REFLEX_LOG_EVERY_SEC", "5.0"))

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OBS_TOPIC = os.getenv("OBS_TOPIC", "vision/observation")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")


class ReflexAgent:
    def __init__(self) -> None:
        self.adapter = build_strategy_state_adapter()
        self.policy = ReflexPolicyAdapter()
        self.client = mqtt.Client(client_id="reflex_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.latest_obs = None

    def run(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        last_log = 0.0
        while not stop_event.is_set():
            now = time.time()
            if now - last_log >= LOG_EVERY_SEC:
                logger.info("Reflex agent heartbeat (obs=%s)", "yes" if self.latest_obs else "no")
                last_log = now
            stop_event.wait(POLL_SEC)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(OBS_TOPIC, 0)])
            logger.info("Connected to MQTT, subscribed to %s", OBS_TOPIC)
        else:
            logger.error("MQTT connect failed rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected rc=%s", rc)

    def _on_message(self, client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            logger.exception("Failed to decode observation payload")
            return
        self.latest_obs = payload
        strategy_state = self.adapter.read()
        action = self.policy.predict(payload, strategy_state)
        if action:
            client.publish(ACT_CMD_TOPIC, json.dumps(action))


def _handle_signal(signum, _frame):
    logger.info("Signal %s received", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    ReflexAgent().run()


if __name__ == "__main__":
    main()
