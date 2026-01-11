#!/usr/bin/env python3
"""Cursor tracker that reads vision frames and publishes cursor coordinates."""
from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
ACT_CMD_TOPIC = os.getenv("CURSOR_ACT_TOPIC") or os.getenv("ACT_CMD_TOPIC", "act/cmd")
MIN_AREA = float(os.getenv("CURSOR_MIN_AREA", "30"))
MAX_AREA = float(os.getenv("CURSOR_MAX_AREA", "1800"))
BINARY_THRESHOLD = int(os.getenv("CURSOR_BINARY_THRESHOLD", "240"))
ASPECT_MIN = float(os.getenv("CURSOR_ASPECT_MIN", "0.2"))
ASPECT_MAX = float(os.getenv("CURSOR_ASPECT_MAX", "5.0"))
STRIDE = max(1, int(os.getenv("CURSOR_FRAME_STRIDE", "2")))
LOSS_ANNOUNCE_SEC = float(os.getenv("CURSOR_LOSS_ANNOUNCE_SEC", "2.0"))
BLUR_KERNEL = int(os.getenv("CURSOR_BLUR_KERNEL", "5"))
MORPH_KERNEL = int(os.getenv("CURSOR_MORPH_KERNEL", "3"))
DIST_WEIGHT = float(os.getenv("CURSOR_DISTANCE_WEIGHT", "0.002"))
HSV_S_MAX = int(os.getenv("CURSOR_HSV_S_MAX", "70"))
HSV_V_MIN = int(os.getenv("CURSOR_HSV_V_MIN", "200"))
MOTION_WEIGHT = float(os.getenv("CURSOR_MOTION_WEIGHT", "2.0"))
MOTION_MIN = float(os.getenv("CURSOR_MOTION_MIN", "4.0"))
MAX_JUMP_NORM = float(os.getenv("CURSOR_MAX_JUMP", "0.2"))
FRAME_QUEUE_SIZE = int(os.getenv("CURSOR_QUEUE_SIZE", "2"))
HEALTH_FILE = os.getenv("CURSOR_HEALTH_FILE", "/tmp/cursor_tracker_status.json")
HEALTH_INTERVAL_SEC = float(os.getenv("CURSOR_HEALTH_INTERVAL_SEC", "5.0"))
HEALTH_OK_SEC = float(os.getenv("CURSOR_HEALTH_OK_SEC", "5.0"))
HEALTH_FAIL_ERROR_SEC = float(os.getenv("CURSOR_HEALTH_FAIL_ERROR_SEC", "15.0"))

# --- Setup ---
logging.basicConfig(level=os.getenv("CURSOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("cursor_tracker")
stop_event = threading.Event()


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


def _prepare_kernel(size: int) -> Tuple[int, int]:
    return (size | 1, size | 1)


def decode_frame(message: dict) -> Optional[np.ndarray]:
    data = get_frame_bytes(message)
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


@dataclass
class CursorDetection:
    x_norm: float
    y_norm: float
    area: float
    bbox: Tuple[int, int, int, int]


def detect_cursor(
    frame: np.ndarray,
    last_point: Optional[Tuple[float, float]],
    prev_gray: Optional[np.ndarray],
) -> Tuple[Optional[CursorDetection], Optional[np.ndarray]]:
    if frame is None or frame.size == 0:
        return None, prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_KERNEL > 1:
        gray = cv2.GaussianBlur(gray, _prepare_kernel(BLUR_KERNEL), 0)
    motion_map = None
    if prev_gray is not None and prev_gray.shape == gray.shape:
        motion_map = cv2.absdiff(gray, prev_gray)
        if BLUR_KERNEL > 1:
            motion_map = cv2.GaussianBlur(motion_map, _prepare_kernel(BLUR_KERNEL), 0)
    _, mask = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    if MORPH_KERNEL > 1:
        kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, gray
    best = None
    best_score = float("-inf")
    height, width = frame.shape[:2]
    last_norm = None
    if last_point is not None and width > 0 and height > 0:
        last_norm = (last_point[0] / width, last_point[1] / height)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect = w / h
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv)
        if HSV_V_MIN > 0 and mean_v < HSV_V_MIN:
            continue
        if HSV_S_MAX > 0 and mean_s > HSV_S_MAX:
            continue
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        if last_norm is not None and MAX_JUMP_NORM > 0:
            dx_norm = abs(cx / width - last_norm[0])
            dy_norm = abs(cy / height - last_norm[1])
            if dx_norm > MAX_JUMP_NORM or dy_norm > MAX_JUMP_NORM:
                continue
        motion_score = 0.0
        if motion_map is not None:
            roi_motion = motion_map[y : y + h, x : x + w]
            if roi_motion.size:
                motion_score = float(np.mean(roi_motion))
        if motion_map is not None and MOTION_MIN > 0 and motion_score < MOTION_MIN:
            continue
        score = area + MOTION_WEIGHT * motion_score
        if last_point is not None:
            dx = cx - last_point[0]
            dy = cy - last_point[1]
            score -= DIST_WEIGHT * (dx * dx + dy * dy)
        if score > best_score:
            best_score = score
            best = (cx, cy, area, (x, y, w, h))
    if not best:
        return None, gray
    cx, cy, area, bbox = best
    detection = CursorDetection(
        x_norm=max(0.0, min(1.0, cx / width)),
        y_norm=max(0.0, min(1.0, cy / height)),
        area=area,
        bbox=bbox,
    )
    return detection, gray


class CursorTrackerAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="cursor_tracker", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self._state_lock = threading.Lock()
        self.last_detection_ts = 0.0
        self.last_publish_ts = 0.0
        self.last_frame_ts = 0.0
        self.last_error_ts = 0.0
        self.connected = False
        self.last_point_px: Optional[Tuple[float, float]] = None
        self.synthetic_point_px: Optional[Tuple[float, float]] = None
        self.synthetic_ts = 0.0
        self.frame_count = 0
        self.loss_notified = False
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_width = 0
        self.frame_height = 0
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self.last_health = None

    def start(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.worker_thread.start()
        self.status_thread.start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Cursor tracker shut down.")

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            with self._state_lock:
                self.connected = True
            topics = [(FRAME_TOPIC, 0)]
            if ACT_CMD_TOPIC:
                topics.append((ACT_CMD_TOPIC, 0))
            client.subscribe(topics)
            client.publish(
                CURSOR_TOPIC,
                json.dumps({"ok": True, "event": "cursor_tracker_ready"}),
            )
            logger.info("Cursor tracker connected.")
        else:
            logger.error("Connect failed: rc=%s", _as_int(rc))
            client.publish(
                CURSOR_TOPIC,
                json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}),
            )

    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0:
            logger.warning("Unexpected disconnect: rc=%s", _as_int(rc))
        with self._state_lock:
            self.connected = False

    def _publish_detection(self, detection: CursorDetection):
        payload = {
            "ok": True,
            "x_norm": round(detection.x_norm, 4),
            "y_norm": round(detection.y_norm, 4),
            "area": round(detection.area, 2),
            "timestamp": time.time(),
        }
        self.client.publish(CURSOR_TOPIC, json.dumps(payload))
        with self._state_lock:
            self.last_publish_ts = payload["timestamp"]

    def _publish_loss(self, reason: str):
        payload = {"ok": False, "reason": reason, "timestamp": time.time()}
        self.client.publish(CURSOR_TOPIC, json.dumps(payload))
        with self._state_lock:
            self.last_point_px = None
            self.last_publish_ts = payload["timestamp"]

    def _on_message(self, client, _userdata, msg):
        if msg.topic == ACT_CMD_TOPIC:
            try:
                data = json.loads(msg.payload.decode("utf-8", "ignore"))
                self._handle_command(data)
            except Exception:
                pass
            return

        if msg.topic == FRAME_TOPIC:
            with self._state_lock:
                self.last_frame_ts = time.time()
            self.frame_count += 1
            if self.frame_count % STRIDE != 0:
                return
            try:
                self.frame_queue.put_nowait(msg.payload)
            except queue.Full:
                pass

    def _worker_loop(self):
        while not stop_event.is_set():
            try:
                payload = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                data = json.loads(payload.decode("utf-8", "ignore"))
                frame = decode_frame(data)
                if frame is None:
                    continue
                
                with self._state_lock:
                    self.frame_height, self.frame_width = frame.shape[:2]
                with self._state_lock:
                    last_point = self.last_point_px
                detection, gray = detect_cursor(frame, last_point, self.prev_gray)
                if gray is not None:
                    self.prev_gray = gray
                
                if detection:
                    now = time.time()
                    with self._state_lock:
                        self.last_detection_ts = now
                        self.last_point_px = (
                            detection.x_norm * frame.shape[1],
                            detection.y_norm * frame.shape[0],
                        )
                        self.synthetic_point_px = self.last_point_px
                        self.synthetic_ts = now
                        self.loss_notified = False
                    self._publish_detection(detection)
                else:
                    fallback = self._synthetic_detection()
                    if fallback:
                        with self._state_lock:
                            self.last_detection_ts = time.time()
                        self._publish_detection(fallback)
                    else:
                        with self._state_lock:
                            self.synthetic_point_px = None
                            self.synthetic_ts = 0.0
                    
                    with self._state_lock:
                        loss_notified = self.loss_notified
                        last_detection_ts = self.last_detection_ts
                    if (not loss_notified and last_detection_ts and 
                        time.time() - last_detection_ts >= LOSS_ANNOUNCE_SEC):
                        with self._state_lock:
                            self.loss_notified = True
                        self._publish_loss("not_found")
            except Exception as e:
                with self._state_lock:
                    self.last_error_ts = time.time()
                logger.error("Worker error: %s", e)

    def _synthetic_detection(self) -> Optional[CursorDetection]:
        with self._state_lock:
            synthetic_point = self.synthetic_point_px
            synthetic_ts = self.synthetic_ts
            width = self.frame_width or 1
            height = self.frame_height or 1
        if not synthetic_point:
            return None
        if time.time() - synthetic_ts > 1.0:
            return None
        x_px = max(0.0, min(float(width), synthetic_point[0]))
        y_px = max(0.0, min(float(height), synthetic_point[1]))
        return CursorDetection(
            x_norm=x_px / max(1.0, width),
            y_norm=y_px / max(1.0, height),
            area=0.0,
            bbox=(int(x_px), int(y_px), 1, 1),
        )

    def _handle_command(self, data: Dict[str, object]):
        action = str(data.get("action") or "").lower()
        if action != "mouse_move":
            return
        try:
            dx = float(data.get("dx", 0.0))
            dy = float(data.get("dy", 0.0))
        except (TypeError, ValueError):
            return
        with self._state_lock:
            if not self.frame_width or not self.frame_height:
                return
            if self.synthetic_point_px is None:
                if self.last_point_px is None:
                    return
                self.synthetic_point_px = tuple(self.last_point_px)
            sx, sy = self.synthetic_point_px
            self.synthetic_point_px = (sx + dx, sy + dy)
            self.synthetic_ts = time.time()

    def _compute_health_snapshot(
        self,
        now: float,
        connected: bool,
        last_error_ts: float,
        last_publish_ts: float,
        last_frame_ts: float,
    ) -> str:
        if not connected:
            return "fail"
        if last_error_ts and now - last_error_ts <= HEALTH_FAIL_ERROR_SEC:
            return "fail"
        last_ok_ts = max(last_publish_ts, last_frame_ts)
        if last_ok_ts and now - last_ok_ts <= HEALTH_OK_SEC:
            return "ok"
        return "degraded"

    def _write_health(self, now: float) -> None:
        with self._state_lock:
            connected = self.connected
            last_frame_ts = self.last_frame_ts
            last_publish_ts = self.last_publish_ts
            last_error_ts = self.last_error_ts
        status = {
            "ok": True,
            "health": self._compute_health_snapshot(
                now,
                connected,
                last_error_ts,
                last_publish_ts,
                last_frame_ts,
            ),
            "connected": connected,
            "last_frame_ts": last_frame_ts or None,
            "last_publish_ts": last_publish_ts or None,
            "last_error_ts": last_error_ts or None,
            "timestamp": now,
        }
        if status["health"] != self.last_health:
            logger.info("Health status: %s", status["health"])
            self.last_health = status["health"]
        try:
            path = Path(HEALTH_FILE)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(status))
            tmp_path.replace(path)
        except Exception as exc:
            logger.warning("Health write failed: %s", exc)

    def _status_loop(self):
        while not stop_event.is_set():
            self._write_health(time.time())
            time.sleep(max(0.5, HEALTH_INTERVAL_SEC))


def _handle_signal(signum, frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = CursorTrackerAgent()
    agent.start()


if __name__ == "__main__":
    main()
