#!/usr/bin/env python3
"""Perception agent that fuses YOLO11 detections + OCR into GameObservation."""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt

sys.path.extend(path for path in ("/app", "/workspace") if path and path not in sys.path)

from core.observations import DetectedObject, GameObservation, OcrZone
from vision.detector_factory import build_detector_backend
from vision.ocr_backends import NullOcrBackend, build_ocr_backend
from vision.perception import PerceptionPipeline
from vision.player_locator import PlayerLocator, PlayerLocatorConfig
from vision.regions import TEXT_REGIONS
from utils.frame_transport import get_frame_bytes

# --- Setup ---
logging.basicConfig(level=os.getenv("PERCEPTION_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("perception_agent")
stop_event = threading.Event()
DEBUG = os.getenv("PERCEPTION_DEBUG", "0").lower() not in {"0", "false", "no"}

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
OBS_TOPIC = os.getenv("PERCEPTION_TOPIC", "vision/observation")
YOLO_WEIGHTS = os.getenv("YOLO11_WEIGHTS", "/mnt/ssd/models/yolo/yolo11n.pt")
YOLO_DEVICE = os.getenv("YOLO11_DEVICE", "cuda:0")
YOLO_CONF = float(os.getenv("YOLO11_CONF", "0.35"))
YOLO_IMGSZ = int(os.getenv("YOLO11_IMGSZ", "640"))
YOLO_MAX_SIZE = int(os.getenv("YOLO_MAX_SIZE", "0"))
YOLO_FALLBACK_CPU = os.getenv("YOLO_FALLBACK_CPU", "0").lower() not in {"0", "false", "no"}
YOLO_CLIP_CPU = os.getenv("YOLO_CLIP_CPU", "1").lower() not in {"0", "false", "no"}
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo11_torch")
YOLO_WORLD_CLASSES = [token.strip() for token in os.getenv("YOLO_WORLD_CLASSES", "enemy,boss,player,npc,loot,portal,waypoint,quest_marker,dialog_button").split(",") if token.strip()]
YOLO_CLASS_LIST = os.getenv("YOLO_CLASS_LIST", "")
YOLO_CLASS_PATH = os.getenv("YOLO_CLASS_PATH", "")
OCR_BACKEND = os.getenv("OCR_BACKEND", "easyocr")
OCR_LANGS = [token.strip() for token in os.getenv("OCR_LANGS", "en").split(",") if token.strip()]
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "1").lower() not in {"0", "false", "no"}
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.05"))
OCR_PADDLE_LANG = os.getenv("OCR_PADDLE_LANG", "")
OCR_USE_ANGLE = os.getenv("OCR_USE_ANGLE_CLS", "1").lower() not in {"0", "false", "no"}
PLAYER_LOCATOR_ENABLED = os.getenv("PLAYER_LOCATOR_ENABLED", "1").lower() not in {"0", "false", "no"}
PLAYER_CENTER_X = float(os.getenv("PLAYER_LOCATOR_CENTER_X", "0.5"))
PLAYER_CENTER_Y = float(os.getenv("PLAYER_LOCATOR_CENTER_Y", "0.55"))
PLAYER_MAX_OFFSET = float(os.getenv("PLAYER_LOCATOR_MAX_OFFSET", "0.45"))
PLAYER_MIN_AREA = float(os.getenv("PLAYER_LOCATOR_MIN_AREA", "0.01"))
PLAYER_ROI = os.getenv("PLAYER_LOCATOR_ROI", "0.25,0.25,0.75,0.9")
PLAYER_MIN_ROI_AREA = float(os.getenv("PLAYER_LOCATOR_MIN_ROI_AREA", "0.02"))

def _load_class_names(class_list: str, class_paths: str) -> List[str]:
    if class_list:
        names = [token.strip() for token in class_list.split(",") if token.strip()]
        if names:
            logger.info("Loaded %d class names from YOLO_CLASS_LIST", len(names))
            return names

    if class_paths:
        for raw in class_paths.split(","):
            cleaned = raw.strip()
            if not cleaned:
                continue
            path = Path(cleaned)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        names = [line.strip() for line in handle if line.strip()]
                    if names:
                        logger.info("Loaded %d class names from %s", len(names), path)
                        return names
                except Exception as exc:
                    logger.warning("Failed to read class names from %s: %s", path, exc)
    return []

YOLO_CLASS_NAMES = _load_class_names(YOLO_CLASS_LIST, YOLO_CLASS_PATH)

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

def decode_frame(message: dict) -> np.ndarray | None:
    data = get_frame_bytes(message)
    if not data:
        return None
    try:
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as exc:
        if DEBUG:
            logger.exception("decode_frame failed: %s", exc)
        return None

def objects_to_dict(objs: List[DetectedObject]) -> List[Dict[str, object]]:
    out = []
    for obj in objs:
        out.append({"label": obj.label, "confidence": round(obj.confidence, 4), "bbox": [round(c, 4) for c in obj.bbox], "extra": obj.extra or {}})
    return out

def object_to_dict(obj: Optional[DetectedObject]) -> Optional[Dict[str, object]]:
    if obj is None: return None
    return {"label": obj.label, "confidence": round(obj.confidence, 4), "bbox": [round(c, 4) for c in obj.bbox], "extra": obj.extra or {}}

def text_zones_to_dict(zones: Dict[str, OcrZone]) -> Dict[str, Dict[str, object]]:
    out = {}
    for name, zone in zones.items():
        out[name] = {"text": zone.text, "confidence": round(zone.confidence, 4), "bbox": [round(c, 4) for c in zone.bbox] if zone.bbox else None}
    return out

class _NullDetector:
    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None):
        return []

class PerceptionAgent:
    def __init__(self) -> None:
        logger.info(
            "Starting perception backend=%s weights=%s device=%s conf=%.3f imgsz=%s max_size=%s classes=%s class_names=%s",
            DETECTOR_BACKEND,
            YOLO_WEIGHTS,
            YOLO_DEVICE,
            YOLO_CONF,
            YOLO_IMGSZ,
            YOLO_MAX_SIZE,
            YOLO_WORLD_CLASSES,
            len(YOLO_CLASS_NAMES),
        )
        self.pipeline = None
        self.pipeline_lock = threading.Lock()
        self.frame_id = 0
        self.client = mqtt.Client(client_id="perception_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.lock = threading.Lock()

    def _init_detector_backend(self):
        try:
            return build_detector_backend(
                DETECTOR_BACKEND,
                weights_path=YOLO_WEIGHTS,
                device=YOLO_DEVICE,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                engine_path=YOLO_WEIGHTS,
                classes=YOLO_WORLD_CLASSES,
                max_size=YOLO_MAX_SIZE,
                fallback_cpu=YOLO_FALLBACK_CPU,
                clip_cpu=YOLO_CLIP_CPU,
                class_names=YOLO_CLASS_NAMES,
            )
        except Exception as exc:
            logger.error("Detector backend error: %s", exc)
            if DEBUG:
                logger.exception("Detector backend stack")
            return _NullDetector()

    def _init_ocr_backend(self):
        languages, paddle_lang = OCR_LANGS or ["en"], OCR_PADDLE_LANG.strip() or (OCR_LANGS or ["en"])[0]
        try:
            backend = build_ocr_backend(OCR_BACKEND, languages=languages, gpu=OCR_USE_GPU, lang=paddle_lang, use_angle_cls=OCR_USE_ANGLE, min_confidence=OCR_MIN_CONF)
            logger.info("OCR backend ready: %s (languages=%s)", OCR_BACKEND, languages)
            return backend
        except Exception as exc:
            logger.error("OCR backend error: %s", exc)
            if DEBUG:
                logger.exception("OCR backend stack")
        return NullOcrBackend()

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return
        with self.pipeline_lock:
            if self.pipeline is not None:
                return
            detector = self._init_detector_backend()
            ocr = self._init_ocr_backend()
            player_locator = self._init_player_locator()
            self.pipeline = PerceptionPipeline(detector=detector, ocr=ocr, text_regions=TEXT_REGIONS, player_locator=player_locator)

    def _init_player_locator(self) -> Optional[PlayerLocator]:
        if not PLAYER_LOCATOR_ENABLED: return None
        try:
            roi_tokens = [float(part.strip()) for part in PLAYER_ROI.split(",")]
            if len(roi_tokens) != 4: raise ValueError
            roi_tuple = tuple(roi_tokens)
        except Exception: roi_tuple = (0.25, 0.25, 0.75, 0.9)
        cfg = PlayerLocatorConfig(center_x=PLAYER_CENTER_X, center_y=PLAYER_CENTER_Y, max_offset=PLAYER_MAX_OFFSET, min_area=PLAYER_MIN_AREA, roi_box=roi_tuple, min_roi_area=PLAYER_MIN_ROI_AREA)
        return PlayerLocator(cfg)

    def run(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Perception agent shut down")

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(FRAME_TOPIC, 0)])
            self.client.publish(OBS_TOPIC, json.dumps({"ok": True, "event": "perception_agent_ready"}))
            logger.info("Connected to MQTT")
        else:
            logger.error("Connect failed rc=%s", _as_int(rc))
            self.client.publish(OBS_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0: logger.warning("Disconnected rc=%s", _as_int(rc))

    def _on_message(self, client, _userdata, msg):
        try: data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            logger.exception("Failed to decode frame message")
            return
        frame = decode_frame(data)
        if frame is None:
            if DEBUG:
                logger.warning("Received empty/invalid frame payload")
            return
        with self.lock:
            self.frame_id += 1
            frame_id = self.frame_id
        try:
            self._ensure_pipeline()
            if self.pipeline is None:
                logger.error("Perception pipeline not ready, dropping frame_id=%s", frame_id)
                return
            observation = self.pipeline.build_observation(frame, frame_id=frame_id)
        except Exception:
            logger.exception("Detection pipeline failed for frame_id=%s", frame_id)
            return
        payload = self._observation_to_payload(observation)
        if DEBUG:
            logger.info(
                "frame_id=%s detections=%d texts=%d",
                frame_id,
                len(payload.get("yolo_objects") or []),
                len(payload.get("text_zones") or {}),
            )
        client.publish(OBS_TOPIC, json.dumps(payload))

    def _observation_to_payload(self, obs: GameObservation) -> Dict[str, object]:
        return {
            "ok": True, "frame_id": obs.frame_id, "yolo_objects": objects_to_dict(obs.yolo_objects),
            "text_zones": text_zones_to_dict(obs.text_zones), "player_candidate": object_to_dict(obs.player_candidate),
            "timestamp": time.time()
        }

def _handle_signal(signum, frame):
    logger.info("Signal %s received", signum)
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = PerceptionAgent()
    agent.run()

if __name__ == "__main__":
    main()
