#!/usr/bin/env python3
"""Hint client: sends frames to external vision server for heavy open-vocab detection."""
from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
import requests

from utils.frame_transport import get_frame_b64

logging.basicConfig(level=os.getenv("HINT_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vision_hint_client")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("FRAME_TOPIC", os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview"))
HINT_TOPIC = os.getenv("HINT_TOPIC", "vision/hints")
HINT_SERVER_URL = os.getenv("HINT_SERVER_URL", "")
HINT_INTERVAL = float(os.getenv("HINT_INTERVAL", "2.0"))
HINT_TIMEOUT = float(os.getenv("HINT_TIMEOUT", "4.0"))
HINT_MAX_QUEUE = int(os.getenv("HINT_MAX_QUEUE", "1"))
HINT_ENABLED = os.getenv("HINT_ENABLED", "1").lower() not in {"0", "false", "no"}


class HintClient:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="hint_client", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(1, HINT_MAX_QUEUE))
        self.stop_event = threading.Event()
        self.last_sent = 0.0

    # MQTT callbacks
    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(FRAME_TOPIC, 0)])
            logger.info("Connected to MQTT")
        else:
            logger.error("MQTT connect failed rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT unexpected disconnect rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            logger.exception("Failed to decode frame message")
            return
        if not payload:
            return
        # keep only the latest frame in queue
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    # Networking
    def _post_frame(self, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        frame_b64 = get_frame_b64(frame)
        if not frame_b64:
            return None
        data = {
            "image_b64": frame_b64,
            "frame_id": frame.get("frame_id"),
            "frame_ts": frame.get("timestamp"),
        }
        try:
            resp = requests.post(HINT_SERVER_URL, json=data, timeout=HINT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hint request failed: %s", exc)
            return None

    def _publish_hint(self, hint: Dict[str, Any]) -> None:
        payload = {
            "ok": bool(hint),
            "timestamp": time.time(),
            "hint": hint,
            "source": "hint_client",
            "server": HINT_SERVER_URL,
        }
        self.client.publish(HINT_TOPIC, json.dumps(payload))

    def run(self):
        if not HINT_ENABLED or not HINT_SERVER_URL:
            logger.warning("Hint client disabled (HINT_ENABLED=%s, HINT_SERVER_URL=%s)", HINT_ENABLED, HINT_SERVER_URL)
            return
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        logger.info("Hint client started: server=%s interval=%.2fs", HINT_SERVER_URL, HINT_INTERVAL)
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                now = time.time()
                if now - self.last_sent < HINT_INTERVAL:
                    continue
                result = self._post_frame(frame)
                if result is not None:
                    self._publish_hint(result)
                self.last_sent = time.time()
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Hint client stopped")


def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down", signum)
    raise SystemExit(0)


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    client = HintClient()
    client.run()


if __name__ == "__main__":
    main()
