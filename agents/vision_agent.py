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

from utils.latency import emit_latency, get_sla_ms

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
MQTT_TOPIC_FRAME = os.environ.get("VISION_FRAME_TOPIC", "vision/frame")
FRAME_PUBLISH_INTERVAL = float(os.environ.get("VISION_FRAME_INTERVAL", "0.5"))
FRAME_JPEG_QUALITY = int(os.environ.get("VISION_FRAME_JPEG_QUALITY", "70"))
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
    
    def _open_capture(self):
        candidates = [VIDEO_DEVICE] + VIDEO_FALLBACKS
        for cand_str in candidates:
            if not cand_str: continue
            cand = int(cand_str) if cand_str.isdigit() else cand_str
            cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
            if cap.isOpened():
                self._configure_capture(cap)
                return cap, cand
            cap.release()
        return None, None

    def _configure_capture(self, cap):
        if VIDEO_WIDTH > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        if VIDEO_HEIGHT > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        if VIDEO_FPS > 0: cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        if VIDEO_PIXFMT:
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*VIDEO_PIXFMT))
            except Exception: pass

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()

        cap, active_device = self._open_capture()
        if cap is None:
            logger.critical(f"Failed to open any video capture device. Searched: {[VIDEO_DEVICE] + VIDEO_FALLBACKS}")
            return
            
        logger.info(f"Capture opened on device {active_device} ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS):.2f}fps)")
        
        last_status_pub = 0
        try:
            while not stop_event.is_set():
                capture_start = time.perf_counter()
                ok, frame = cap.read()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0
                if not ok or frame is None:
                    stop_event.wait(0.02)
                    continue
                emit_latency(
                    self.client,
                    "capture",
                    capture_ms,
                    sla_ms=SLA_STAGE_CAPTURE_MS,
                    tags={"device": str(active_device)},
                    agent="vision_agent",
                )

                with self.frame_lock:
                    self.last_frame = frame

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
                now = time.time()
                self.client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": True, "mean": round(float(np.mean(gray)), 2), "timestamp": now}), qos=0)

                with self.config_lock:
                    frame_interval = self.current_frame_interval
                
                if FRAME_STREAM_ENABLED and (now - self.last_frame_pub_ts) >= frame_interval:
                    self.last_frame_pub_ts = now
                    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(FRAME_JPEG_QUALITY, 95))])
                    if ok:
                        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
                        self.client.publish(MQTT_TOPIC_FRAME, json.dumps({"ok": True, "timestamp": now, "image_b64": b64, "width": frame.shape[1], "height": frame.shape[0]}), qos=0)

                if VISION_STATUS_TOPIC and (now - last_status_pub) >= VISION_STATUS_INTERVAL:
                    self.publish_status()
                    last_status_pub = now
                
                stop_event.wait(0.05)
        finally:
            cap.release()
            self.client.loop_stop()
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
