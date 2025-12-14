#!/usr/bin/env python3
"""MQTT object-detection agent backed by pluggable YOLO detectors."""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt

logger = logging.getLogger("object_detection_agent")
logging.basicConfig(level=os.getenv("OBJECT_LOG_LEVEL", "INFO"))

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame")
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC", "vision/objects")
MODEL_PATH = os.getenv("OBJECT_MODEL_PATH", "")
DETECTOR_BACKEND = os.getenv("OBJECT_DETECTOR_BACKEND", "dummy").lower()
DEVICE = os.getenv("OBJECT_DEVICE", "cpu")
CONF_THRESHOLD = float(os.getenv("OBJECT_CONF_THRESHOLD", "0.35"))
IOU_THRESHOLD = float(os.getenv("OBJECT_IOU_THRESHOLD", "0.45"))
FRAME_QUEUE_MAX = int(os.getenv("OBJECT_QUEUE", "2"))
IMG_SIZE = int(os.getenv("OBJECT_IMAGE_SIZE", "640"))
VISION_CONFIG_TOPIC = os.getenv("VISION_CONFIG_TOPIC", "vision/config")
VISION_MODE_DEFAULT = os.getenv("VISION_MODE_DEFAULT", "medium").lower()

MODE_SETTINGS = {
    "low": {
        "frame_stride": 3,
        "conf": min(0.6, CONF_THRESHOLD + 0.15),
        "imgsz": max(320, IMG_SIZE // 2),
    },
    "medium": {
        "frame_stride": 2,
        "conf": CONF_THRESHOLD,
        "imgsz": IMG_SIZE,
    },
    "high": {
        "frame_stride": 1,
        "conf": max(0.1, CONF_THRESHOLD - 0.1),
        "imgsz": max(IMG_SIZE, 640),
    },
}


@dataclass
class Detection:
    """Structured detection result published to MQTT."""

    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

    def as_dict(self) -> dict:
        return {"class": self.label, "confidence": round(self.confidence, 4), "box": [round(v, 2) for v in self.box]}


def decode_frame(image_b64: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG string into a BGR numpy array."""

    if not image_b64:
        return None
    try:
        buffer = base64.b64decode(image_b64)
        array = np.frombuffer(buffer, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:  # pragma: no cover - OpenCV specific errors
        logger.warning("failed to decode frame: %s", exc)
        return None


class BaseDetector:
    """Interface for concrete detectors."""

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def update_runtime(self, _config: dict):  # pragma: no cover - optional override
        return


class DummyDetector(BaseDetector):
    """Fallback detector that emits no objects."""

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        return []


class UltralyticsDetector(BaseDetector):
    """Detector that uses ultralytics.YOLO checkpoints."""

    def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.35, imgsz: int = 640):
        if not model_path:
            raise ValueError("OBJECT_MODEL_PATH must be set for Ultralytics backend")
        from ultralytics import YOLO  # Imported lazily to keep startup light when unused

        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
        self._imgsz = imgsz

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model.predict(source=rgb, device=self._device, conf=self._conf, imgsz=self._imgsz, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names or getattr(self._model, "names", {})
        for box, cls_idx, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = names.get(int(cls_idx), f"cls_{int(cls_idx)}") if isinstance(names, dict) else str(int(cls_idx))
            detections.append(Detection(label=label, confidence=float(conf), box=[float(v) for v in box]))
        return detections

    def update_runtime(self, config: dict):
        self._conf = float(config.get("conf", self._conf))
        self._imgsz = int(config.get("imgsz", self._imgsz))


def build_detector() -> BaseDetector:
    """Factory that instantiates the requested detector backend."""

    if DETECTOR_BACKEND == "ultralytics":
        try:
            return UltralyticsDetector(MODEL_PATH, DEVICE, CONF_THRESHOLD, IMG_SIZE)
        except Exception as exc:  # pragma: no cover - missing deps handled at runtime
            logger.error("Failed to init Ultralytics detector: %s", exc)
    else:
        logger.info("Object detector backend %s not recognized, using dummy", DETECTOR_BACKEND)
    return DummyDetector()


class ObjectDetectionAgent:
    """Consumes frames from vision agent and publishes structured detections."""

    def __init__(self):
        self.client = mqtt.Client(client_id="object_detection_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.detector = build_detector()
        self.vision_mode = VISION_MODE_DEFAULT if VISION_MODE_DEFAULT in MODE_SETTINGS else "medium"
        settings = MODE_SETTINGS.get(self.vision_mode, MODE_SETTINGS["medium"])
        self.frame_stride = settings.get("frame_stride", 1)
        self._frame_counter = 0
        self._apply_detector_settings(settings)

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = [(FRAME_TOPIC, 0)]
            if VISION_CONFIG_TOPIC:
                topics.append((VISION_CONFIG_TOPIC, 0))
            client.subscribe(topics)
            logger.info("Subscribed to %s", FRAME_TOPIC)
            client.publish(OBJECT_TOPIC, json.dumps({"ok": True, "event": "object_detector_ready"}))
        else:
            logger.error("MQTT connect failed: rc=%s", rc)
            client.publish(OBJECT_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected: rc=%s", rc)

    def _on_message(self, client, _userdata, msg):
        if msg.topic == VISION_CONFIG_TOPIC:
            self._handle_config(msg.payload)
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            logger.debug("non-JSON frame payload dropped")
            return

        self._frame_counter += 1
        if self.frame_stride > 1 and self._frame_counter % self.frame_stride != 0:
            return

        frame = decode_frame(payload.get("image_b64", ""))
        if frame is None:
            return
        detections = self.detector.detect(frame)
        payload = {
            "ok": True,
            "timestamp": time.time(),
            "frame_ts": payload.get("timestamp", time.time()),
            "objects": [det.as_dict() for det in detections],
        }
        self.client.publish(OBJECT_TOPIC, json.dumps(payload))

    def _apply_detector_settings(self, settings: dict):
        if hasattr(self.detector, "update_runtime"):
            self.detector.update_runtime(settings)

    def _handle_config(self, payload: bytes):
        try:
            data = json.loads(payload.decode("utf-8", "ignore"))
        except Exception:
            return
        mode = str(data.get("mode") or data.get("vision_mode") or "").lower()
        if mode not in MODE_SETTINGS:
            return
        self.vision_mode = mode
        settings = MODE_SETTINGS[mode]
        self.frame_stride = max(1, settings.get("frame_stride", 1))
        self._apply_detector_settings(settings)
        logger.info("vision mode switched to %s (stride=%s)", mode, self.frame_stride)


def main():
    agent = ObjectDetectionAgent()
    agent.start()


if __name__ == "__main__":
    main()
