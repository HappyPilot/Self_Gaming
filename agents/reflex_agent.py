#!/usr/bin/env python3
"""Reflex agent that consumes observations and strategy state."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time

import json
try:
    import paho.mqtt.client as mqtt
except Exception:  # noqa: BLE001
    mqtt = None

from shared_state.strategy_state import build_strategy_state_adapter
from agents.reflex_policy import ReflexPolicyAdapter

logging.basicConfig(level=os.getenv("REFLEX_LOG_LEVEL", "INFO"))
logger = logging.getLogger("reflex_agent")
stop_event = threading.Event()

POLL_SEC = float(os.getenv("REFLEX_POLL_SEC", "0.1"))
LOG_EVERY_SEC = float(os.getenv("REFLEX_LOG_EVERY_SEC", "5.0"))

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OBS_TOPIC = os.getenv("OBS_TOPIC", os.getenv("SCENE_TOPIC", "scene/state"))
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")


class ReflexAgent:
    def __init__(self) -> None:
        self.adapter = build_strategy_state_adapter()
        self.policy = ReflexPolicyAdapter()
        self.mqtt_available = mqtt is not None
        self.client = None
        if self.mqtt_available:
            self.client = mqtt.Client(client_id="reflex_agent", protocol=mqtt.MQTTv311)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
        self.latest_obs = None

    def run(self) -> None:
        if not self.mqtt_available or self.client is None:
            logger.error("paho-mqtt not available; reflex_agent exiting")
            return
        try:
            self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to connect to MQTT: %s", exc)
            return
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
        if not isinstance(payload, dict):
            logger.warning("Ignoring non-dict observation payload")
            return
        self.latest_obs = payload
        strategy_state = self.adapter.read()
        try:
            action = self.policy.predict(payload, strategy_state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reflex policy error: %s", exc)
            return
        if action:
            try:
                client.publish(ACT_CMD_TOPIC, json.dumps(action))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to publish reflex action: %s", exc)


def _handle_signal(signum, _frame):
    logger.info("Signal %s received", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    ReflexAgent().run()


if __name__ == "__main__":
    main()
