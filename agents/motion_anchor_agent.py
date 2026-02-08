#!/usr/bin/env python3
"""Motion anchor agent: publishes motion-based anchors from vision frames."""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes
from utils.motion_anchor import compute_motion_center

try:
    import vpi  # type: ignore
    _VPI_AVAILABLE = True
except Exception:
    vpi = None
    _VPI_AVAILABLE = False


MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
ANCHOR_TOPIC = os.getenv("MOTION_ANCHOR_TOPIC", "vision/anchors")
ANCHOR_ENABLED = os.getenv("MOTION_ANCHOR_ENABLE", "1") != "0"
BACKEND_PREFERENCE = os.getenv("MOTION_BACKEND", "vpi")
DOWNSCALE_W = int(os.getenv("MOTION_DOWNSCALE_W", "160"))
DOWNSCALE_H = int(os.getenv("MOTION_DOWNSCALE_H", "90"))
MAG_THRESHOLD = float(os.getenv("MOTION_MAG_THRESHOLD", "5.0"))
MIN_MEAN = float(os.getenv("MOTION_MIN_MEAN", "0.5"))
MASK_BOTTOM = float(os.getenv("MOTION_UI_MASK_BOTTOM", "0.2"))
MASK_CORNER_X = float(os.getenv("MOTION_UI_MASK_CORNER_X", "0.2"))
PUBLISH_COOLDOWN = float(os.getenv("MOTION_PUBLISH_COOLDOWN", "0.1"))
FRAME_QUEUE_SIZE = int(os.getenv("MOTION_FRAME_QUEUE", "2"))
LOG_LEVEL = os.getenv("MOTION_ANCHOR_LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("motion_anchor")
stop_event = threading.Event()


def build_anchor_payload(x_norm: float, y_norm: float, score: float, ts: float) -> dict:
    return {
        "label": "motion_salient",
        "point": [round(float(x_norm), 4), round(float(y_norm), 4)],
        "score": float(score),
        "ts": float(ts),
    }


def resolve_backend(prefer: str, *, vpi_available: bool) -> str:
    prefer = (prefer or "").strip().lower()
    if prefer == "vpi" and vpi_available:
        return "vpi"
    return "cpu"


def _decode_frame(message: dict) -> Optional[np.ndarray]:
    data = get_frame_bytes(message)
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _resize_gray(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_w, target_h = size
    if target_w > 0 and target_h > 0:
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return gray


class MotionAnchorAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="motion_anchor")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.backend = resolve_backend(BACKEND_PREFERENCE, vpi_available=_VPI_AVAILABLE)
        self.vpi_failed = False
        self.prev_gray: Optional[np.ndarray] = None
        self.queue: "queue.Queue[dict]" = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.last_publish_ts = 0.0

    def start(self) -> None:
        if not ANCHOR_ENABLED:
            logger.warning("Motion anchor disabled via env")
            return
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.worker.start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(FRAME_TOPIC, 0)])
            logger.info("Motion anchor connected; subscribed to %s", FRAME_TOPIC)
        else:
            logger.error("Motion anchor failed to connect: rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        logger.warning("Motion anchor disconnected rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if msg.topic != FRAME_TOPIC:
            return
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                pass

    def _compute_motion_vpi(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if not _VPI_AVAILABLE or self.vpi_failed:
            return None
        try:
            v1 = vpi.asimage(prev_gray, vpi.Format.U8)
            v2 = vpi.asimage(curr_gray, vpi.Format.U8)
            flow = vpi.optflow_dense(v1, v2, backend=vpi.Backend.OFA, quality=vpi.OptFlowQuality.MEDIUM)
            with flow.lock_cpu() as data:
                arr = np.array(data)
            if arr.ndim < 3 or arr.shape[-1] < 2:
                return None
            dx = arr[..., 0].astype(np.float32)
            dy = arr[..., 1].astype(np.float32)
            mag = np.sqrt(dx * dx + dy * dy)
            zeros = np.zeros_like(mag, dtype=np.float32)
            return compute_motion_center(
                zeros,
                mag,
                mag_threshold=MAG_THRESHOLD,
                min_mean=MIN_MEAN,
                mask_bottom=MASK_BOTTOM,
                mask_corner_x=MASK_CORNER_X,
            )
        except Exception as exc:
            logger.warning("VPI motion failed; falling back to CPU: %s", exc)
            self.vpi_failed = True
            return None

    def _compute_motion_cpu(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        return compute_motion_center(
            prev_gray,
            curr_gray,
            mag_threshold=MAG_THRESHOLD,
            min_mean=MIN_MEAN,
            mask_bottom=MASK_BOTTOM,
            mask_corner_x=MASK_CORNER_X,
        )

    def _worker_loop(self) -> None:
        while not stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            frame = _decode_frame(payload)
            if frame is None:
                continue
            gray = _resize_gray(frame, (DOWNSCALE_W, DOWNSCALE_H))
            if self.prev_gray is None:
                self.prev_gray = gray
                continue
            center = None
            if self.backend == "vpi":
                center = self._compute_motion_vpi(self.prev_gray, gray)
                if center is None:
                    self.backend = "cpu"
            if self.backend == "cpu":
                center = self._compute_motion_cpu(self.prev_gray, gray)
            self.prev_gray = gray
            if not center:
                continue
            now = time.time()
            if now - self.last_publish_ts < PUBLISH_COOLDOWN:
                continue
            x_norm, y_norm, score = center
            payload = build_anchor_payload(x_norm, y_norm, score, ts=now)
            self.client.publish(ANCHOR_TOPIC, json.dumps(payload))
            self.last_publish_ts = now


def main() -> None:
    agent = MotionAnchorAgent()
    agent.start()


if __name__ == "__main__":
    main()
