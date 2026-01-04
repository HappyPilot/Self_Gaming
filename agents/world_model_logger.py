#!/usr/bin/env python3
"""Log embedding transitions for world-model training."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=os.getenv("WORLD_MODEL_LOGGER_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("world_model_logger")
stop_event = threading.Event()

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EMBED_TOPIC = os.getenv("WORLD_MODEL_EMBED_TOPIC", os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings"))
ACTION_TOPIC = os.getenv("WORLD_MODEL_ACTION_TOPIC", os.getenv("ACT_CMD_TOPIC", "act/cmd"))
REWARD_TOPIC = os.getenv("WORLD_MODEL_REWARD_TOPIC", os.getenv("REWARD_TOPIC", "train/reward"))
STATUS_TOPIC = os.getenv("WORLD_MODEL_LOGGER_STATUS_TOPIC", "world_model_logger/status")
DATASET_DIR = Path(os.getenv("WORLD_MODEL_DATASET_DIR", "/mnt/ssd/datasets/world_model"))
SESSION_ID = os.getenv("WORLD_MODEL_SESSION_ID", "").strip()
REQUIRE_REWARD = os.getenv("WORLD_MODEL_REQUIRE_REWARD", "0") not in {"0", "false", "False"}
MAX_EMBED_DIM = int(os.getenv("WORLD_MODEL_MAX_EMBED_DIM", "4096"))
MAX_ACTION_AGE_SEC = float(os.getenv("WORLD_MODEL_MAX_ACTION_AGE_SEC", "5.0"))
MAX_RECORD_BYTES = int(os.getenv("WORLD_MODEL_MAX_RECORD_BYTES", "0"))


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_embedding(payload: dict) -> Optional[list]:
    if not isinstance(payload, dict):
        return None
    embedding = payload.get("embedding") or payload.get("embeddings")
    if not isinstance(embedding, list):
        return None
    if not embedding:
        return None
    if MAX_EMBED_DIM and len(embedding) > MAX_EMBED_DIM:
        logger.warning("Embedding dim too large (%s > %s); dropping.", len(embedding), MAX_EMBED_DIM)
        return None
    return embedding


class WorldModelLogger:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="world_model_logger", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.lock = threading.Lock()
        self.last_embedding: Optional[dict] = None
        self.pending_action: Optional[dict] = None
        self.last_reward: Optional[dict] = None

        session_id = SESSION_ID or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.base_dir = DATASET_DIR / session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.base_dir / "transitions.jsonl"
        self.output_fp = self.output_path.open("a", encoding="utf-8", buffering=1)
        meta = {
            "session_id": session_id,
            "created_at": time.time(),
            "embed_topic": EMBED_TOPIC,
            "action_topic": ACTION_TOPIC,
            "reward_topic": REWARD_TOPIC,
            "require_reward": REQUIRE_REWARD,
            "max_embed_dim": MAX_EMBED_DIM,
            "max_action_age_sec": MAX_ACTION_AGE_SEC,
        }
        (self.base_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("World model logger writing to %s", self.output_path)

    def start(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self.output_fp.close()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = [(t, 0) for t in {EMBED_TOPIC, ACTION_TOPIC, REWARD_TOPIC} if t]
            client.subscribe(topics)
            client.publish(
                STATUS_TOPIC,
                json.dumps({"ok": True, "event": "world_model_logger_ready", "path": str(self.output_path)}),
            )
            logger.info("Connected to MQTT, subscribed to %s", [t for t, _ in topics])
        else:
            logger.error("MQTT connect failed: rc=%s", rc)
            client.publish(
                STATUS_TOPIC,
                json.dumps({"ok": False, "event": "connect_failed", "code": rc}),
            )

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected: rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if msg.topic == EMBED_TOPIC:
            self._handle_embedding(payload)
        elif msg.topic == ACTION_TOPIC:
            self._handle_action(payload)
        elif msg.topic == REWARD_TOPIC:
            self._handle_reward(payload)

    def _handle_action(self, payload: dict) -> None:
        if not isinstance(payload, dict) or "action" not in payload:
            return
        ts = _as_float(payload.get("timestamp"), time.time())
        with self.lock:
            self.pending_action = {"payload": payload, "timestamp": ts}

    def _handle_reward(self, payload: dict) -> None:
        if not isinstance(payload, dict) or "reward" not in payload:
            return
        ts = _as_float(payload.get("timestamp"), time.time())
        try:
            reward_value = float(payload.get("reward"))
        except (TypeError, ValueError):
            return
        with self.lock:
            self.last_reward = {
                "reward": reward_value,
                "timestamp": ts,
                "stage": payload.get("stage"),
                "components": payload.get("components"),
            }

    def _handle_embedding(self, payload: dict) -> None:
        embedding = _extract_embedding(payload)
        if embedding is None:
            return
        frame_ts = _as_float(payload.get("frame_ts"), None)
        ts = _as_float(payload.get("timestamp"), time.time())
        emb_ts = frame_ts if frame_ts is not None else ts
        current = {
            "embedding": embedding,
            "timestamp": emb_ts,
            "backend": payload.get("backend"),
            "dim": len(embedding),
            "frame_id": payload.get("frame_id"),
        }
        with self.lock:
            self._maybe_emit_transition(current)
            self.last_embedding = current

    def _maybe_emit_transition(self, current: dict) -> None:
        if self.last_embedding is None or self.pending_action is None:
            return
        action_ts = self.pending_action.get("timestamp")
        if action_ts is None:
            return
        if MAX_ACTION_AGE_SEC > 0 and current["timestamp"] - action_ts > MAX_ACTION_AGE_SEC:
            logger.debug("Dropping stale action (age %.2fs)", current["timestamp"] - action_ts)
            self.pending_action = None
            return
        if action_ts < self.last_embedding["timestamp"]:
            return
        if action_ts > current["timestamp"]:
            return
        if REQUIRE_REWARD and self.last_reward is None:
            return
        record = {
            "timestamp": time.time(),
            "emb_t": self.last_embedding["embedding"],
            "emb_t_ts": self.last_embedding["timestamp"],
            "emb_t1": current["embedding"],
            "emb_t1_ts": current["timestamp"],
            "action": self.pending_action["payload"],
            "action_ts": action_ts,
            "reward": self.last_reward,
            "backend": current.get("backend"),
            "dim": current.get("dim"),
        }
        self._write_record(record)
        self.pending_action = None

    def _write_record(self, record: dict) -> None:
        payload = json.dumps(record)
        if MAX_RECORD_BYTES and len(payload) > MAX_RECORD_BYTES:
            logger.warning("Transition payload too large (%s bytes), dropping.", len(payload))
            return
        try:
            self.output_fp.write(payload + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write transition: %s", exc)


def _handle_signal(signum, _frame):
    logger.info("Signal %s received, shutting down.", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    WorldModelLogger().start()


if __name__ == "__main__":
    main()
