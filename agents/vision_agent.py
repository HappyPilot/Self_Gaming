#!/usr/bin/env python3
import os
import json
import base64
import time
import signal
import threading
import logging

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from capture import build_capture_backend
from utils.latency import emit_latency, get_sla_ms
from utils.frame_transport import SHM_AVAILABLE, ShmFrameRing, get_transport_mode

# --- Setup ---
logging.basicConfig(level=os.getenv("VISION_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vision_agent")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST = os.environ.get("MQTT_HOST", "mq")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_CMD = os.environ.get("MQTT_TOPIC_CMD", "vision/cmd")
MQTT_TOPIC_METRIC = os.environ.get("MQTT_TOPIC_METRIC", "vision/mean")
MQTT_TOPIC_SNAPSHOT = os.environ.get("MQTT_TOPIC_SNAPSHOT", "vision/snapshot")
MQTT_TOPIC_FRAME = os.environ.get("VISION_FRAME_TOPIC", "vision/frame/preview")
MQTT_TOPIC_FRAME_PREVIEW = os.environ.get("VISION_FRAME_PREVIEW_TOPIC", MQTT_TOPIC_FRAME)
MQTT_TOPIC_FRAME_FULL = os.environ.get("VISION_FRAME_FULL_TOPIC", "vision/frame/full")
FRAME_PUBLISH_INTERVAL = float(os.environ.get("VISION_FRAME_INTERVAL", "0.5"))
FRAME_JPEG_QUALITY = int(os.environ.get("VISION_FRAME_JPEG_QUALITY", "70"))
FRAME_PREVIEW_QUALITY = int(os.environ.get("VISION_FRAME_PREVIEW_QUALITY", FRAME_JPEG_QUALITY))
FRAME_FULL_QUALITY = int(os.environ.get("VISION_FRAME_FULL_QUALITY", "95"))
FRAME_TRANSPORT = get_transport_mode()
CAPTURE_BACKEND = os.environ.get("CAPTURE_BACKEND", "v4l2")
FRAME_SHM_MAX_BYTES = int(os.environ.get("FRAME_SHM_MAX_BYTES", str(5 * 1024 * 1024)))
FRAME_SHM_SLOTS = int(os.environ.get("FRAME_SHM_SLOTS", "8"))
FRAME_SHM_PREFIX = os.environ.get("FRAME_SHM_PREFIX", "sg_frame")
FRAME_STREAM_ENABLED = os.environ.get("VISION_FRAME_ENABLED", "1") not in {"0", "false", "False"}
VISION_CONFIG_TOPIC = os.environ.get("VISION_CONFIG_TOPIC", "vision/config")
VISION_STATUS_TOPIC = os.environ.get("VISION_STATUS_TOPIC", "vision/status")
VISION_MODE_DEFAULT = os.environ.get("VISION_MODE_DEFAULT", "medium").lower()
VISION_STATUS_INTERVAL = float(os.environ.get("VISION_STATUS_INTERVAL", "15"))
VIDEO_DEVICE = os.environ.get("VIDEO_DEVICE", "/dev/video0")
VIDEO_FALLBACKS = [p.strip() for p in os.environ.get("VIDEO_DEVICE_FALLBACKS", "/dev/video2,/dev/video1,0,2").split(",") if p.strip()]
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "0"))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "0"))
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "0"))
VIDEO_PIXFMT = os.environ.get("VIDEO_PIXFMT", "")[:4]
SLA_STAGE_CAPTURE_MS = get_sla_ms("SLA_STAGE_CAPTURE_MS")

def _float_env(name: str, fallback: float) -> float:
    try: return float(os.environ.get(name, str(fallback)))
    except (TypeError, ValueError): return fallback

MODE_FRAME_INTERVALS = {
    "low": _float_env("VISION_FRAME_INTERVAL_LOW", FRAME_PUBLISH_INTERVAL * 1.5 or 0.8),
    "medium": _float_env("VISION_FRAME_INTERVAL_MED", FRAME_PUBLISH_INTERVAL),
    "high": _float_env("VISION_FRAME_INTERVAL_HIGH", max(0.1, FRAME_PUBLISH_INTERVAL * 0.5)),
}

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class VisionAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="vision", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self.config_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.last_frame: np.ndarray | None = None
        self.last_frame_pub_ts = 0.0
        self.current_mode = VISION_MODE_DEFAULT if VISION_MODE_DEFAULT in MODE_FRAME_INTERVALS else "medium"
        self.current_frame_interval = MODE_FRAME_INTERVALS[self.current_mode]
        self.last_config_reason = "default"
        self.frame_transport = FRAME_TRANSPORT
        self._last_shm_fallback_log = 0.0
        self.shm_ring = None
        if self.frame_transport == "shm":
            if not SHM_AVAILABLE:
                logger.warning("FRAME_TRANSPORT=shm requested but shared_memory is unavailable; falling back to mqtt")
                self.frame_transport = "mqtt"
            else:
                self.shm_ring = ShmFrameRing(FRAME_SHM_MAX_BYTES, FRAME_SHM_SLOTS, FRAME_SHM_PREFIX)
                logger.info(
                    "SHM transport enabled (slots=%s, max_bytes=%s, prefix=%s)",
                    FRAME_SHM_SLOTS,
                    FRAME_SHM_MAX_BYTES,
                    FRAME_SHM_PREFIX,
                )

    def _build_frame_payload(self, frame: np.ndarray, quality: int, variant: str, timestamp: float) -> dict | None:
        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(quality, 95))],
        )
        if not ok:
            return None
        jpg_bytes = jpg.tobytes()
        payload = {
            "ok": True,
            "timestamp": timestamp,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "variant": variant,
        }
        if self.frame_transport == "shm" and self.shm_ring:
            desc = self.shm_ring.write(jpg_bytes)
            if not desc:
                now = time.time()
                if now - self._last_shm_fallback_log >= 5.0:
                    logger.warning(
                        "SHM write failed (size=%s, max=%s); falling back to mqtt",
                        len(jpg_bytes),
                        self.shm_ring.max_bytes,
                    )
                    self._last_shm_fallback_log = now
                return self._build_base64_payload(payload, jpg_bytes)
            payload.update(desc)
            payload["transport"] = "shm"
            payload["encoding"] = "jpeg"
            return payload
        return self._build_base64_payload(payload, jpg_bytes)

    def _build_base64_payload(self, payload: dict, jpg_bytes: bytes) -> dict:
        b64 = base64.b64encode(jpg_bytes).decode("ascii")
        payload["image_b64"] = b64
        return payload

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            topics = [(MQTT_TOPIC_CMD, 0), (VISION_CONFIG_TOPIC, 0)] if VISION_CONFIG_TOPIC else [(MQTT_TOPIC_CMD, 0)]
            client.subscribe(topics)
            logger.info("Connected to MQTT and subscribed to: %s", [t for t,_ in topics])
        else:
            logger.error(f"Failed to connect to MQTT: rc={_as_int(rc)}")

    def _on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from MQTT: rc={_as_int(rc)}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", "ignore")
            if msg.topic == VISION_CONFIG_TOPIC:
                self._apply_vision_mode(json.loads(payload))
                return
            
            data = json.loads(payload) if payload.startswith("{") else {"cmd": payload.strip()}
            cmd = (data.get("cmd") or data.get("action") or "").lower()
            
            if cmd == "ping":
                self.client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": True, "pong": True}))
            elif cmd == "snapshot":
                self._handle_snapshot()
            elif cmd in ("stop", "quit", "exit"):
                stop_event.set()
            else:
                self.client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": False, "error": "unknown_cmd", "cmd": cmd}))
        except Exception as e:
            logger.error(f"Error in on_message: {e}")

    def _handle_snapshot(self):
        with self.frame_lock:
            frame = self.last_frame
        if frame is None:
            self.client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": False, "error": "no_frame"}))
            return
        
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            self.client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": False, "error": "encode_fail"}))
            return
            
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
        self.client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": True, "image_b64": b64}))

    def _apply_vision_mode(self, config: dict):
        mode = str(config.get("mode") or config.get("vision_mode") or "").lower()
        if mode not in MODE_FRAME_INTERVALS:
            return
        with self.config_lock:
            self.current_mode = mode
            self.current_frame_interval = MODE_FRAME_INTERVALS[mode]
            self.last_config_reason = config.get("reason", "config")
        self.publish_status()

    def publish_status(self):
        if not VISION_STATUS_TOPIC: return
        with self.config_lock:
            payload = {"ok": True, "event": "vision_status", "mode": self.current_mode,
                       "frame_interval": round(self.current_frame_interval, 3), "reason": self.last_config_reason}
        try:
            self.client.publish(VISION_STATUS_TOPIC, json.dumps(payload), qos=0, retain=False)
        except Exception as e:
            logger.warning(f"Failed to publish status: {e}")

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()

        capture = build_capture_backend(CAPTURE_BACKEND)
        if not capture.start():
            logger.error("Failed to start capture backend %s; falling back to dummy", CAPTURE_BACKEND)
            capture = build_capture_backend("dummy")
            if not capture.start():
                logger.critical("Failed to start dummy capture backend")
                return
        logger.info("Capture backend started: %s (%s)", capture.name, capture.describe())
        
        last_status_pub = 0
        try:
            while not stop_event.is_set():
                capture_start = time.perf_counter()
                handle = capture.read()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0
                if handle is None:
                    stop_event.wait(0.02)
                    continue
                frame = handle.to_numpy()
                emit_latency(
                    self.client,
                    "capture",
                    capture_ms,
                    sla_ms=SLA_STAGE_CAPTURE_MS,
                    tags={"device": str(handle.source or capture.name)},
                    agent="vision_agent",
                )

                with self.frame_lock:
                    self.last_frame = frame

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
                now = handle.timestamp
                self.client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": True, "mean": round(float(np.mean(gray)), 2), "timestamp": now}), qos=0)

                with self.config_lock:
                    frame_interval = self.current_frame_interval
                
                if FRAME_STREAM_ENABLED and (now - self.last_frame_pub_ts) >= frame_interval:
                    self.last_frame_pub_ts = now
                    preview_topic = MQTT_TOPIC_FRAME_PREVIEW or ""
                    full_topic = MQTT_TOPIC_FRAME_FULL or ""
                    legacy_topic = MQTT_TOPIC_FRAME or ""
                    preview_topics = []
                    if preview_topic:
                        preview_topics.append(preview_topic)
                    if legacy_topic and legacy_topic not in preview_topics and legacy_topic != full_topic:
                        preview_topics.append(legacy_topic)
                    if preview_topics:
                        payload = self._build_frame_payload(frame, FRAME_PREVIEW_QUALITY, "preview", now)
                        if payload:
                            packed = json.dumps(payload)
                            for topic in preview_topics:
                                self.client.publish(topic, packed, qos=0)
                    if full_topic and full_topic not in preview_topics:
                        payload = self._build_frame_payload(frame, FRAME_FULL_QUALITY, "full", now)
                        if payload:
                            self.client.publish(full_topic, json.dumps(payload), qos=0)

                if VISION_STATUS_TOPIC and (now - last_status_pub) >= VISION_STATUS_INTERVAL:
                    self.publish_status()
                    last_status_pub = now
                
                stop_event.wait(0.05)
        finally:
            capture.stop()
            self.client.loop_stop()
            if self.shm_ring:
                self.shm_ring.close()
            self.client.disconnect()
            logger.info("Vision agent shut down cleanly.")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received, shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = VisionAgent()
    agent.run()

if __name__ == "__main__":
    main()
