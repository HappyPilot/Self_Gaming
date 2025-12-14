#!/usr/bin/env python3
"""Automatic controller that adjusts vision modes based on thermal + model error."""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import deque
from typing import Deque, Optional

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
THERMAL_TOPIC = os.getenv("THERMAL_TOPIC", "system/thermal")
PRED_ERROR_TOPIC = os.getenv("PRED_ERROR_TOPIC", "world_model/pred_error")
VISION_CONFIG_TOPIC = os.getenv("VISION_CONFIG_TOPIC", "vision/config")

VISION_ORDER = ["low", "medium", "high"]
DEFAULT_MODE = os.getenv("VISION_MODE_DEFAULT", "medium").lower()
if DEFAULT_MODE not in VISION_ORDER:
    DEFAULT_MODE = "medium"

ERROR_HIGH = float(os.getenv("VISION_CTRL_ERROR_HIGH", "0.7"))
ERROR_LOW = float(os.getenv("VISION_CTRL_ERROR_LOW", "0.3"))
SWITCH_COOLDOWN = float(os.getenv("VISION_CTRL_SWITCH_COOLDOWN_SEC", "15"))
ERROR_WINDOW = int(os.getenv("VISION_CTRL_ERROR_WINDOW", "6"))
TICK_INTERVAL = float(os.getenv("VISION_CTRL_TICK", "0.5"))

LOG_LEVEL = os.getenv("VISION_CTRL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("vision_controller")

# TODO: Integrate reward/goal quality (train/reward) to bias mode selection when objectives demand it.
# TODO: Allow manual override via a dedicated flag in vision/config for research toggles.
# TODO: Add hooks for ROI-based sampling once vision/roi feeds are available.


def _mode_index(mode: str) -> int:
    try:
        return VISION_ORDER.index(mode)
    except ValueError:
        return VISION_ORDER.index("medium")


class VisionControllerAgent:
    """Simple hysteresis controller that toggles vision modes for Jetson."""

    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="vision_controller", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.error_history: Deque[float] = deque(maxlen=max(3, ERROR_WINDOW))
        self.current_mode = DEFAULT_MODE
        self.thermal_state = "ok"
        self.last_switch = 0.0
        self.stop = False

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = []
            if PRED_ERROR_TOPIC:
                topics.append((PRED_ERROR_TOPIC, 0))
            if THERMAL_TOPIC:
                topics.append((THERMAL_TOPIC, 0))
            if topics:
                client.subscribe(topics)
            logger.info(
                "vision_controller connected (mode=%s, err_high=%.2f, err_low=%.2f, cooldown=%.1fs)",
                self.current_mode,
                ERROR_HIGH,
                ERROR_LOW,
                SWITCH_COOLDOWN,
            )
        else:
            logger.error("vision_controller failed to connect rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("vision_controller disconnected unexpectedly rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == PRED_ERROR_TOPIC:
            self._handle_pred_error(msg.payload)
        elif msg.topic == THERMAL_TOPIC:
            self._handle_thermal(msg.payload)

    def _handle_pred_error(self, payload: bytes):
        try:
            data = json.loads(payload.decode("utf-8", "ignore"))
        except json.JSONDecodeError:
            return
        value = data.get("error")
        if isinstance(value, (int, float)):
            self.error_history.append(float(value))

    def _handle_thermal(self, payload: bytes):
        try:
            data = json.loads(payload.decode("utf-8", "ignore"))
        except json.JSONDecodeError:
            return
        state = str(data.get("state", "")).lower()
        if state in {"ok", "warm", "hot"}:
            self.thermal_state = state

    def _avg_error(self) -> Optional[float]:
        if not self.error_history:
            return None
        return sum(self.error_history) / len(self.error_history)

    def _thermal_cap(self) -> str:
        if self.thermal_state == "hot":
            return "low"
        if self.thermal_state == "warm":
            return "medium"
        return "high"

    def _publish_mode(self, mode: str, reason: str, avg_error: Optional[float]):
        if not VISION_CONFIG_TOPIC:
            return
        payload = {
            "mode": mode,
            "reason": reason,
            "thermal": self.thermal_state,
            "avg_error": round(avg_error, 4) if avg_error is not None else None,
            "source": "vision_controller",
            "timestamp": time.time(),
        }
        self.client.publish(VISION_CONFIG_TOPIC, json.dumps(payload))
        logger.info(
            "vision mode %s -> %s due to %s (thermal=%s avg_err=%s)",
            self.current_mode,
            mode,
            reason,
            self.thermal_state,
            f"{avg_error:.3f}" if avg_error is not None else "n/a",
        )
        self.current_mode = mode
        self.last_switch = time.time()

    def _maybe_switch(self):
        now = time.time()
        if now - self.last_switch < SWITCH_COOLDOWN:
            return
        avg_error = self._avg_error()
        cap_mode = self._thermal_cap()
        desired = self.current_mode
        reason = None

        if cap_mode == "low" and self.current_mode != "low":
            desired, reason = "low", "thermal_hot"
        elif cap_mode == "medium" and _mode_index(self.current_mode) > _mode_index("medium"):
            desired, reason = "medium", "thermal_warm"

        if reason is None and avg_error is not None:
            idx = _mode_index(self.current_mode)
            cap_idx = _mode_index(cap_mode)
            if avg_error > ERROR_HIGH and idx < cap_idx:
                desired = VISION_ORDER[min(idx + 1, cap_idx)]
                if desired != self.current_mode:
                    reason = "high_error"
            elif avg_error < ERROR_LOW and idx > 0:
                desired = VISION_ORDER[max(0, idx - 1)]
                if desired != self.current_mode:
                    reason = "low_error"

        if reason and desired != self.current_mode:
            self._publish_mode(desired, reason, avg_error)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        try:
            while not self.stop:
                self._maybe_switch()
                time.sleep(TICK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("vision_controller received KeyboardInterrupt, shutting down")
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def main():
    agent = VisionControllerAgent()

    def _handle_signal(_sig, _frame):
        agent.stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    agent.run()


if __name__ == "__main__":
    main()

