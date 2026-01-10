#!/usr/bin/env python3
"""Experience logger that writes embedding transitions to disk."""
from __future__ import annotations

import array
import base64
import gzip
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
    level=os.getenv("EXP_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("experience_logger")
stop_event = threading.Event()

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACTION_TOPIC = os.getenv("EXP_ACTION_TOPIC", os.getenv("ACT_CMD_TOPIC", "act/cmd"))
ACTION_ALIAS = os.getenv("EXP_ACTION_ALIAS_TOPIC", "").strip()
REWARD_TOPIC = os.getenv("EXP_REWARD_TOPIC", os.getenv("REWARD_TOPIC", "train/reward"))

EXP_LOG_ENABLED = os.getenv("EXP_LOG_ENABLED", "1") not in {"0", "false", "False"}
EXP_LOG_DIR = Path(os.getenv("EXP_LOG_DIR", "/mnt/ssd/datasets/rollouts"))
EXP_LOG_FORMAT = os.getenv("EXP_LOG_FORMAT", "jsonl").strip().lower()
EXP_LOG_MAX_MB = float(os.getenv("EXP_LOG_MAX_MB", "200"))
EXP_LOG_FLUSH_SEC = float(os.getenv("EXP_LOG_FLUSH_SEC", "2"))
EXP_MATCH_WINDOW_MS = int(os.getenv("EXP_MATCH_WINDOW_MS", "500"))
EXP_MAX_WAIT_MS = int(os.getenv("EXP_MAX_WAIT_MS", "2000"))
EXP_STATUS_INTERVAL_SEC = float(os.getenv("EXP_STATUS_INTERVAL_SEC", "30"))
EXP_STORE_EMB_BYTES = os.getenv("EXP_STORE_EMB_BYTES", "0") not in {"0", "false", "False"}
EXP_LOG_COMPRESS = os.getenv("EXP_LOG_COMPRESS", os.getenv("EXPERIENCE_COMPRESS", "0")) not in {"0", "false", "False"}
EXP_LOG_MAX_RECORDS = int(os.getenv("EXP_LOG_MAX_RECORDS", "0"))
EXP_SESSION_ID = os.getenv("EXP_LOG_SESSION_ID", "").strip()
EXP_GAME_ID = (os.getenv("EXP_LOG_GAME_ID") or os.getenv("RECORDER_GAME_ID") or os.getenv("GAME_ID") or "default").strip()
EXP_REQUIRE_REWARD = os.getenv("EXP_REQUIRE_REWARD", "0") not in {"0", "false", "False"}
EXP_REWARD_DEFAULT = float(os.getenv("EXP_REWARD_DEFAULT", "0.0"))
EXP_LOG_STATUS_TOPIC = os.getenv("EXP_LOG_STATUS_TOPIC", "experience_logger/status")

MAX_EMBED_DIM = int(os.getenv("EXP_MAX_EMBED_DIM", "4096"))


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_embedding(scene: Dict[str, Any]) -> Optional[list]:
    embedding = scene.get("embeddings") or scene.get("embedding")
    if not isinstance(embedding, list):
        return None
    if not embedding:
        return None
    if MAX_EMBED_DIM and len(embedding) > MAX_EMBED_DIM:
        logger.warning("Embedding dim too large (%s > %s); dropping.", len(embedding), MAX_EMBED_DIM)
        return None
    return embedding


def _encode_embedding_bytes(embedding: list) -> str:
    arr = array.array("f", [float(x) for x in embedding])
    return base64.b64encode(arr.tobytes()).decode("ascii")


class RollingJsonlWriter:
    def __init__(self) -> None:
        self.max_bytes = int(EXP_LOG_MAX_MB * 1024 * 1024) if EXP_LOG_MAX_MB > 0 else 0
        self.max_records = EXP_LOG_MAX_RECORDS if EXP_LOG_MAX_RECORDS > 0 else 0
        self.flush_sec = EXP_LOG_FLUSH_SEC
        self.compress = EXP_LOG_COMPRESS
        self.records_written = 0
        self.bytes_written = 0
        self.last_flush = time.time()
        self.chunk_idx = 0
        self.base_dir = EXP_LOG_DIR / EXP_GAME_ID
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = EXP_SESSION_ID or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.handle = None
        self._open_new()
        meta = {
            "session_id": self.session_id,
            "game_id": EXP_GAME_ID,
            "created_at": time.time(),
            "scene_topic": SCENE_TOPIC,
            "action_topic": ACTION_TOPIC,
            "reward_topic": REWARD_TOPIC,
            "require_reward": EXP_REQUIRE_REWARD,
            "format": EXP_LOG_FORMAT,
            "compress": self.compress,
            "max_mb": EXP_LOG_MAX_MB,
            "flush_sec": EXP_LOG_FLUSH_SEC,
            "match_window_ms": EXP_MATCH_WINDOW_MS,
            "max_wait_ms": EXP_MAX_WAIT_MS,
            "store_emb_bytes": EXP_STORE_EMB_BYTES,
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _open_new(self) -> None:
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        filename = f"rollout_{self.chunk_idx:04d}{suffix}"
        path = self.session_dir / filename
        if self.compress:
            self.handle = gzip.open(path, "at", encoding="utf-8")
        else:
            self.handle = path.open("a", encoding="utf-8")
        self.chunk_idx += 1
        self.records_written = 0
        self.bytes_written = 0
        logger.info("Experience logger writing to %s", path)

    def _rotate(self) -> None:
        if self.handle:
            try:
                self.handle.flush()
            except Exception:
                pass
            try:
                self.handle.close()
            except Exception:
                pass
        self._open_new()

    def write(self, record: Dict[str, Any]) -> None:
        if self.handle is None:
            self._open_new()
        line = json.dumps(record, separators=(",", ":"))
        try:
            self.handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write record: %s", exc)
            return
        self.records_written += 1
        self.bytes_written += len(line) + 1
        now = time.time()
        if self.flush_sec > 0 and (now - self.last_flush) >= self.flush_sec:
            try:
                self.handle.flush()
            except Exception:
                pass
            self.last_flush = now
        if self._should_rotate():
            self._rotate()

    def _should_rotate(self) -> bool:
        if self.max_bytes and self.bytes_written >= self.max_bytes:
            return True
        if self.max_records and self.records_written >= self.max_records:
            return True
        return False

    def close(self) -> None:
        if self.handle:
            try:
                self.handle.flush()
            except Exception:
                pass
            try:
                self.handle.close()
            except Exception:
                pass
            self.handle = None


class ExperienceLogger:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="experience_logger", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.lock = threading.Lock()
        self.last_embedding: Optional[dict] = None
        self.pending_action: Optional[dict] = None
        self.last_reward: Optional[dict] = None
        self.writer = RollingJsonlWriter() if EXP_LOG_ENABLED else None
        self.connected = False
        self.stats = {
            "transitions_written": 0,
            "actions_dropped_pending": 0,
            "actions_dropped_no_emb": 0,
            "actions_dropped_window": 0,
            "actions_dropped_max_wait": 0,
            "last_write_ts": None,
        }
        self.status_thread = threading.Thread(target=self._status_loop, daemon=True)

        if not EXP_LOG_ENABLED:
            logger.warning("EXP_LOG_ENABLED=0; experience logger will idle.")

    def start(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        if EXP_STATUS_INTERVAL_SEC > 0:
            self.status_thread.start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        if EXP_STATUS_INTERVAL_SEC > 0 and self.status_thread.is_alive():
            self.status_thread.join(timeout=2)
        if self.writer:
            self.writer.close()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = [(SCENE_TOPIC, 0), (ACTION_TOPIC, 0), (REWARD_TOPIC, 0)]
            if ACTION_ALIAS:
                topics.append((ACTION_ALIAS, 0))
            client.subscribe(topics)
            self.connected = True
            client.publish(
                EXP_LOG_STATUS_TOPIC,
                json.dumps({"ok": True, "event": "experience_logger_ready", "dir": str(EXP_LOG_DIR)}),
            )
            logger.info("Connected to MQTT, subscribed to %s", [t for t, _ in topics])
        else:
            logger.error("MQTT connect failed: rc=%s", rc)
            client.publish(
                EXP_LOG_STATUS_TOPIC,
                json.dumps({"ok": False, "event": "connect_failed", "code": rc}),
            )

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected: rc=%s", rc)
        self.connected = False

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if msg.topic == SCENE_TOPIC:
            self._handle_scene(payload)
        elif msg.topic in {ACTION_TOPIC, ACTION_ALIAS}:
            self._handle_action(payload, msg.topic)
        elif msg.topic == REWARD_TOPIC:
            self._handle_reward(payload)

    def _handle_scene(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        embedding = _extract_embedding(payload)
        if embedding is None:
            return
        emb_ts = _as_float(payload.get("embeddings_ts"), None)
        frame_ts = _as_float(payload.get("frame_ts"), None)
        scene_ts = _as_float(payload.get("timestamp"), time.time())
        emb_ts = emb_ts if emb_ts is not None else frame_ts if frame_ts is not None else scene_ts
        current = {
            "embedding": embedding,
            "emb_ts": emb_ts,
            "scene_ts": scene_ts,
            "frame_ts": frame_ts,
            "dim": len(embedding),
            "meta": self._extract_meta(payload),
        }
        with self.lock:
            self._maybe_emit_transition(current)
            self.last_embedding = current

    def _handle_action(self, payload: dict, topic: str) -> None:
        if not isinstance(payload, dict):
            return
        if "action" not in payload and "vector" not in payload:
            return
        action_ts = _as_float(payload.get("timestamp"), time.time())
        with self.lock:
            if self.pending_action is not None:
                logger.debug("Pending action already set; dropping new action.")
                self.stats["actions_dropped_pending"] += 1
                return
            if self.last_embedding is None:
                logger.debug("Action received without embedding; dropping.")
                self.stats["actions_dropped_no_emb"] += 1
                return
            if EXP_MATCH_WINDOW_MS > 0:
                window = EXP_MATCH_WINDOW_MS / 1000.0
                if action_ts < self.last_embedding["emb_ts"] or (action_ts - self.last_embedding["emb_ts"]) > window:
                    logger.debug("Action too far from last embedding; dropping.")
                    self.stats["actions_dropped_window"] += 1
                    return
            action_id = payload.get("action_id") or payload.get("id") or payload.get("request_id")
            action_id = str(action_id) if action_id else str(uuid.uuid4())
            self.pending_action = {
                "payload": payload,
                "timestamp": action_ts,
                "emb_t": self.last_embedding,
                "action_id": action_id,
                "topic": topic,
            }

    def _handle_reward(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        reward_value = payload.get("reward")
        try:
            reward_value = float(reward_value)
        except (TypeError, ValueError):
            return
        reward_ts = _as_float(payload.get("timestamp"), time.time())
        with self.lock:
            self.last_reward = {"value": reward_value, "timestamp": reward_ts, "payload": payload}

    def _maybe_emit_transition(self, current: dict) -> None:
        if not self.writer or not self.pending_action or not self.last_embedding:
            return
        action_ts = self.pending_action.get("timestamp")
        emb_t = self.pending_action.get("emb_t") or self.last_embedding
        if action_ts is None:
            return
        if action_ts < emb_t["emb_ts"]:
            self.stats["actions_dropped_window"] += 1
            self.pending_action = None
            return
        if current["emb_ts"] <= action_ts:
            return
        if EXP_MAX_WAIT_MS > 0:
            max_wait = EXP_MAX_WAIT_MS / 1000.0
            if (current["emb_ts"] - action_ts) > max_wait:
                logger.debug("Next embedding exceeded max wait; dropping action.")
                self.stats["actions_dropped_max_wait"] += 1
                self.pending_action = None
                return
        reward = self._resolve_reward()
        if reward is None:
            return
        record = self._build_record(emb_t, current, reward)
        self.writer.write(record)
        now = time.time()
        self.stats["transitions_written"] += 1
        self.stats["last_write_ts"] = now
        self.pending_action = None

    def _resolve_reward(self) -> Optional[dict]:
        if self.last_reward is None:
            if EXP_REQUIRE_REWARD:
                return None
            return {"value": EXP_REWARD_DEFAULT, "timestamp": time.time(), "source": "default"}
        return {"value": self.last_reward["value"], "timestamp": self.last_reward["timestamp"], "source": "reward_topic"}

    def _build_record(self, emb_t: dict, emb_t1: dict, reward: dict) -> dict:
        action_payload = self.pending_action.get("payload") if self.pending_action else None
        record = {
            "timestamp": time.time(),
            "emb_t_ts": emb_t["emb_ts"],
            "emb_t1_ts": emb_t1["emb_ts"],
            "action_ts": self.pending_action.get("timestamp") if self.pending_action else None,
            "action_id": self.pending_action.get("action_id") if self.pending_action else None,
            "action_topic": self.pending_action.get("topic") if self.pending_action else None,
            "action": action_payload,
            "reward": reward,
            "meta": emb_t1.get("meta", {}),
        }
        if EXP_STORE_EMB_BYTES:
            record["emb_t_bytes"] = _encode_embedding_bytes(emb_t["embedding"])
            record["emb_t1_bytes"] = _encode_embedding_bytes(emb_t1["embedding"])
            record["emb_dim"] = emb_t1.get("dim")
            record["emb_dtype"] = "float32"
        else:
            record["emb_t"] = emb_t["embedding"]
            record["emb_t1"] = emb_t1["embedding"]
        return record

    def _extract_meta(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        objects = scene.get("objects") or []
        texts = scene.get("text") or []
        return {
            "scene_ts": _as_float(scene.get("timestamp"), None),
            "embeddings_ts": _as_float(scene.get("embeddings_ts"), None),
            "frame_ts": _as_float(scene.get("frame_ts"), None),
            "game_id": scene.get("game_id") or EXP_GAME_ID,
            "flags": scene.get("flags"),
            "mean": scene.get("mean"),
            "objects_count": len(objects) if isinstance(objects, list) else None,
            "text_count": len(texts) if isinstance(texts, list) else None,
        }

    def _status_loop(self) -> None:
        while not stop_event.wait(EXP_STATUS_INTERVAL_SEC):
            self._publish_status()

    def _publish_status(self) -> None:
        if not self.connected:
            return
        with self.lock:
            payload = {
                "ok": True,
                "event": "experience_logger_status",
                "timestamp": time.time(),
                **self.stats,
            }
        try:
            self.client.publish(EXP_LOG_STATUS_TOPIC, json.dumps(payload))
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "Experience logger status transitions=%s drops(pending=%s no_emb=%s window=%s max_wait=%s)",
            payload.get("transitions_written"),
            payload.get("actions_dropped_pending"),
            payload.get("actions_dropped_no_emb"),
            payload.get("actions_dropped_window"),
            payload.get("actions_dropped_max_wait"),
        )


def _handle_signal(signum, _frame):
    logger.info("Signal %s received, shutting down.", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    ExperienceLogger().start()


if __name__ == "__main__":
    main()
