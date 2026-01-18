#!/usr/bin/env python3
"""Embedding guard: detect in-game vs non-game based on vision embeddings."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import paho.mqtt.client as mqtt


logging.basicConfig(level=os.getenv("EMBEDDING_GUARD_LOG_LEVEL", "INFO"))
logger = logging.getLogger("embedding_guard")
stop_event = threading.Event()

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EMBED_TOPIC = os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings")
FLAGS_TOPIC = os.getenv("SCENE_FLAGS_TOPIC", "scene/flags")
PAUSE_TOPIC = os.getenv("EMBEDDING_GUARD_PAUSE_TOPIC", "")

MODE = os.getenv("EMBEDDING_GUARD_MODE", "run").lower()
DATA_DIR = Path(os.getenv("EMBEDDING_GUARD_DATA_DIR", "/mnt/ssd/models/embedding_guard"))
GAME_SAMPLES_PATH = Path(os.getenv("EMBEDDING_GUARD_GAME_SAMPLES", str(DATA_DIR / "game_embeddings.json")))
NON_SAMPLES_PATH = Path(os.getenv("EMBEDDING_GUARD_NON_SAMPLES", str(DATA_DIR / "non_embeddings.json")))
MU_GAME_PATH = Path(os.getenv("EMBEDDING_GUARD_MU_GAME", str(DATA_DIR / "mu_game.json")))
MU_NON_PATH = Path(os.getenv("EMBEDDING_GUARD_MU_NON", str(DATA_DIR / "mu_non.json")))

TARGET_SAMPLES = int(os.getenv("EMBEDDING_GUARD_SAMPLES", "40"))
EMBED_DIM = int(os.getenv("EMBEDDING_GUARD_DIM", "768"))
MARGIN = float(os.getenv("EMBEDDING_GUARD_MARGIN", "0.05"))
ENTER_MARGIN = float(os.getenv("EMBEDDING_GUARD_ENTER_MARGIN", str(MARGIN)))
EXIT_MARGIN = float(os.getenv("EMBEDDING_GUARD_EXIT_MARGIN", str(-MARGIN)))
DEBOUNCE_FRAMES = int(os.getenv("EMBEDDING_GUARD_DEBOUNCE", "3"))
PUBLISH_INTERVAL = float(os.getenv("EMBEDDING_GUARD_PUBLISH_INTERVAL", "1.0"))


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return (vec / norm).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _load_samples(path: Path) -> List[np.ndarray]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load samples from %s: %s", path, exc)
        return []
    out = []
    if isinstance(data, list):
        for item in data:
            arr = np.asarray(item, dtype=np.float32).reshape(-1)
            if arr.size == EMBED_DIM:
                out.append(arr)
    return out


def _save_samples(path: Path, samples: List[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.tolist() for sample in samples]
    path.write_text(json.dumps(payload))


def _load_centroid(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        arr = np.asarray(data, dtype=np.float32).reshape(-1)
        if arr.size != EMBED_DIM:
            return None
        return _normalize(arr)
    except Exception as exc:
        logger.warning("Failed to load centroid from %s: %s", path, exc)
        return None


def _save_centroid(path: Path, centroid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(centroid.tolist()))


def _compute_centroid(samples: List[np.ndarray]) -> Optional[np.ndarray]:
    if not samples:
        return None
    normalized = []
    for sample in samples:
        normed = _normalize(sample)
        if normed is not None:
            normalized.append(normed)
    if not normalized:
        return None
    centroid = np.mean(np.stack(normalized, axis=0), axis=0)
    return _normalize(centroid)


class EmbeddingGuardAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="embedding_guard", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.game_samples = _load_samples(GAME_SAMPLES_PATH)
        self.non_samples = _load_samples(NON_SAMPLES_PATH)
        self.mu_game = _load_centroid(MU_GAME_PATH) or _compute_centroid(self.game_samples)
        self.mu_non = _load_centroid(MU_NON_PATH) or _compute_centroid(self.non_samples)
        if self.mu_game is not None and not MU_GAME_PATH.exists():
            _save_centroid(MU_GAME_PATH, self.mu_game)
        if self.mu_non is not None and not MU_NON_PATH.exists():
            _save_centroid(MU_NON_PATH, self.mu_non)

        self.state: Optional[bool] = None
        self.pending_state: Optional[bool] = None
        self.pending_count = 0
        self.last_publish = 0.0

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(EMBED_TOPIC, 0)])
            logger.info("Embedding guard ready: mode=%s topic=%s", MODE, EMBED_TOPIC)
        else:
            logger.error("MQTT connect failed: rc=%s", rc)

    def _on_message(self, client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        embedding = payload.get("embedding") or payload.get("embeddings")
        if not isinstance(embedding, list):
            return
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size != EMBED_DIM:
            logger.debug("Embedding dim mismatch: %s != %s", vec.size, EMBED_DIM)
            return
        self._handle_embedding(vec)

    def _handle_embedding(self, vec: np.ndarray) -> None:
        if MODE.startswith("collect_game"):
            self._collect_sample(vec, self.game_samples, GAME_SAMPLES_PATH, MU_GAME_PATH, "game")
        elif MODE.startswith("collect_non"):
            self._collect_sample(vec, self.non_samples, NON_SAMPLES_PATH, MU_NON_PATH, "non")

        if self.mu_game is None or self.mu_non is None:
            return

        normed = _normalize(vec)
        if normed is None:
            return
        score_game = _cosine(normed, self.mu_game)
        score_non = _cosine(normed, self.mu_non)
        delta = score_game - score_non
        if self.state is True:
            raw_state = delta > EXIT_MARGIN
        else:
            raw_state = delta > ENTER_MARGIN
        self._update_state(raw_state, score_game, score_non, delta)

    def _collect_sample(
        self,
        vec: np.ndarray,
        sample_list: List[np.ndarray],
        sample_path: Path,
        centroid_path: Path,
        label: str,
    ) -> None:
        if len(sample_list) >= TARGET_SAMPLES:
            return
        sample_list.append(vec.astype(np.float32))
        if len(sample_list) % 5 == 0:
            logger.info("Collected %s/%s %s samples", len(sample_list), TARGET_SAMPLES, label)
        if len(sample_list) >= TARGET_SAMPLES:
            _save_samples(sample_path, sample_list)
            centroid = _compute_centroid(sample_list)
            if centroid is not None:
                _save_centroid(centroid_path, centroid)
                if label == "game":
                    self.mu_game = centroid
                else:
                    self.mu_non = centroid
                logger.info("Saved %s centroid to %s", label, centroid_path)

    def _update_state(self, candidate: bool, score_game: float, score_non: float, delta: float) -> None:
        now = time.time()
        if self.state is None:
            self.state = candidate
            self._publish_state(now, score_game, score_non, delta, changed=True)
            return
        if candidate != self.state:
            if self.pending_state != candidate:
                self.pending_state = candidate
                self.pending_count = 1
            else:
                self.pending_count += 1
            if self.pending_count >= DEBOUNCE_FRAMES:
                self.state = candidate
                self.pending_state = None
                self.pending_count = 0
                self._publish_state(now, score_game, score_non, delta, changed=True)
            return

        self.pending_state = None
        self.pending_count = 0
        if (now - self.last_publish) >= PUBLISH_INTERVAL:
            self._publish_state(now, score_game, score_non, delta, changed=False)

    def _publish_state(self, now: float, score_game: float, score_non: float, delta: float, changed: bool) -> None:
        self.last_publish = now
        payload = {
            "ok": True,
            "timestamp": now,
            "source": "embedding_guard",
            "flags": {"in_game": bool(self.state)},
            "scores": {
                "game": round(score_game, 4),
                "non_game": round(score_non, 4),
                "delta": round(delta, 4),
            },
            "changed": changed,
        }
        if FLAGS_TOPIC:
            self.client.publish(FLAGS_TOPIC, json.dumps(payload))
        if PAUSE_TOPIC and self.state is False:
            self.client.publish(
                PAUSE_TOPIC,
                json.dumps(
                    {"action": "wait", "source": "embedding_guard", "reason": "not_in_game", "timestamp": now}
                ),
            )

    def run(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()


def _handle_signal(signum, _frame):
    logger.info("Signal %s received; stopping", signum)
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    EmbeddingGuardAgent().run()


if __name__ == "__main__":
    main()
