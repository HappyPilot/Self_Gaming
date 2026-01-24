#!/usr/bin/env python3
"""MQTT object-detection agent backed by pluggable YOLO detectors."""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# Add local Ultralytics repo (inside image or shared host volume)
ULTRA_PATHS = ["/app/ultralytics_repo", "/opt/ultralytics_repo"]
for ultra_path in ULTRA_PATHS:
    if os.path.isdir(ultra_path) and ultra_path not in sys.path:
        sys.path.append(ultra_path)

from dataclasses import dataclass
from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes
from utils.latency import emit_latency, get_sla_ms
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None
logger = logging.getLogger("object_detection_agent")
logging.basicConfig(level=os.getenv("OBJECT_LOG_LEVEL", "INFO"))

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC") or os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
MODEL_PATH = os.getenv("OBJECT_MODEL_PATH", "")
DETECTOR_BACKEND = os.getenv("OBJECT_DETECTOR_BACKEND", "dummy").lower()
DEVICE = os.getenv("OBJECT_DEVICE", "cpu")
CONF_THRESHOLD = float(os.getenv("OBJECT_CONF_THRESHOLD", "0.35"))
IOU_THRESHOLD = float(os.getenv("OBJECT_IOU_THRESHOLD", "0.45"))
FRAME_QUEUE_MAX = int(os.getenv("OBJECT_QUEUE", "2"))
IMG_SIZE = int(os.getenv("OBJECT_IMAGE_SIZE", "640"))
CLASS_LIST = os.getenv("OBJECT_CLASS_LIST", "")  # Optional: comma-separated names
CLASS_PATHS = os.getenv("OBJECT_CLASS_PATH", "")  # Optional: comma-separated paths to names files
GAME_ID = os.getenv("GAME_ID", "").strip().lower()
FALLBACK_BACKEND = os.getenv("OBJECT_FALLBACK_BACKEND", "").strip().lower()
FALLBACK_MODEL_PATH = os.getenv("OBJECT_FALLBACK_MODEL_PATH", "").strip()
FALLBACK_DEVICE = os.getenv("OBJECT_FALLBACK_DEVICE", "").strip()
FALLBACK_CONF_THRESHOLD = float(os.getenv("OBJECT_FALLBACK_CONF_THRESHOLD", str(CONF_THRESHOLD)))
FALLBACK_IOU_THRESHOLD = float(os.getenv("OBJECT_FALLBACK_IOU_THRESHOLD", str(IOU_THRESHOLD)))
FALLBACK_IMG_SIZE = int(os.getenv("OBJECT_FALLBACK_IMAGE_SIZE", str(IMG_SIZE)))
FALLBACK_CLASS_LIST = os.getenv("OBJECT_FALLBACK_CLASS_LIST", "")
FALLBACK_CLASS_PATHS = os.getenv("OBJECT_FALLBACK_CLASS_PATH", "")
VISION_CONFIG_TOPIC = os.getenv("VISION_CONFIG_TOPIC", "vision/config")
VISION_MODE_DEFAULT = os.getenv("VISION_MODE_DEFAULT", "medium").lower()
SLA_STAGE_DETECT_MS = get_sla_ms("SLA_STAGE_DETECT_MS")

MODE_SETTINGS = {
    "low": {"frame_stride": 4, "conf": min(0.6, CONF_THRESHOLD + 0.15), "imgsz": max(320, IMG_SIZE // 2)},
    "medium": {"frame_stride": 2, "conf": CONF_THRESHOLD, "imgsz": IMG_SIZE},
    "high": {"frame_stride": 1, "conf": max(0.1, CONF_THRESHOLD - 0.1), "imgsz": max(IMG_SIZE, 640)},
}

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


def _load_class_names(
    model_path: str,
    class_list: str = "",
    class_paths: str = "",
    game_id: str = "",
) -> dict[int, str]:
    """
    Resolve class names with precedence:
    1) OBJECT_CLASS_LIST env (comma-separated)
    2) OBJECT_CLASS_PATH env (comma-separated file paths)
    3) Sidecar files near the model (and GAME_ID-specific variants)
    4) Default COCO-ish stub cls_{i}
    """
    # 1) Direct list from env
    if class_list:
        names = [n.strip() for n in class_list.split(",") if n.strip()]
        if names:
            logger.info("Loaded class names from OBJECT_CLASS_LIST (%d classes)", len(names))
            return {i: name for i, name in enumerate(names)}

    # 2) Custom file(s) from env
    if class_paths:
        for raw in class_paths.split(","):
            path = Path(raw.strip())
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        names = [line.strip() for line in f.readlines() if line.strip()]
                    if names:
                        logger.info("Loaded class names from %s (%d classes)", path, len(names))
                        return {i: name for i, name in enumerate(names)}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read class names from %s: %s", path, exc)

    # 3) Sidecars near the model (optionally with GAME_ID suffix)
    try:
        root = Path(model_path).resolve()
        candidates = [
            root.with_suffix(".names"),
            root.with_suffix(".txt"),
            root.parent / "classes.txt",
            root.parent / "names.txt",
        ]
        if game_id:
            candidates.extend([
                root.with_name(f"{root.stem}_{game_id}.names"),
                root.with_name(f"{root.stem}_{game_id}.txt"),
                root.parent / f"classes_{game_id}.txt",
                root.parent / f"names_{game_id}.txt",
            ])
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f.readlines() if line.strip()]
                if names:
                    logger.info("Loaded class names from %s (%d classes)", path, len(names))
                    return {i: name for i, name in enumerate(names)}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load class names: %s", exc)

    # 4) Fallback
    return {i: f"cls_{i}" for i in range(80)}


@dataclass
class Detection:
    label: str
    confidence: float
    box: List[float]

    def as_dict(self) -> dict:
        return {"class": self.label, "confidence": round(self.confidence, 4), "box": [round(v, 2) for v in self.box]}


def decode_frame(image_payload: str | bytes) -> Optional[np.ndarray]:
    if not image_payload:
        return None
    try:
        if isinstance(image_payload, str):
            buffer = base64.b64decode(image_payload)
        else:
            buffer = image_payload
        array = np.frombuffer(buffer, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("failed to decode frame: %s", exc)
        return None


class BaseDetector:
    backend = "base"

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        raise NotImplementedError

    def update_runtime(self, _config: dict):
        pass


class DummyDetector(BaseDetector):
    backend = "dummy"

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        return []


class UltralyticsDetector(BaseDetector):
    backend = "ultralytics"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf: float = 0.35,
        imgsz: int = 640,
        class_list: str = "",
        class_paths: str = "",
    ):
        if not model_path:
            raise ValueError("OBJECT_MODEL_PATH must be set for Ultralytics backend")
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        # Allow overriding names if provided via env or sidecar files.
        self._override_names = _load_class_names(model_path, class_list, class_paths, GAME_ID)

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Force FP32 (half=False) to prevent CUDA/AMP crashes on Jetson
        if torch is not None:
            with torch.no_grad():
                results = self._model.predict(
                    source=rgb,
                    device=self._device,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                    half=False,
                )
        else:
            results = self._model.predict(
                source=rgb,
                device=self._device,
                conf=self._conf,
                imgsz=self._imgsz,
                verbose=False,
                half=False,
            )
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = self._override_names or result.names or getattr(self._model, "names", {})
        for box, cls_idx, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = names.get(int(cls_idx), f"cls_{int(cls_idx)}")
            detections.append(Detection(label=label, confidence=float(conf), box=[float(v) for v in box]))
        return detections

    def update_runtime(self, config: dict):
        self._conf = float(config.get("conf", self._conf))
        self._imgsz = int(config.get("imgsz", self._imgsz))


class OnnxDetector(BaseDetector):
    backend = "onnx"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.35,
        imgsz: int = 640,
        class_list: str = "",
        class_paths: str = "",
    ):
        if not model_path:
            raise ValueError("ONNX model path required")
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed")

        self.ort = ort
        self.model_path = model_path
        self._requested_device = device
        self._conf = conf
        self._imgsz = imgsz
        self._providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if "cuda" in device else ['CPUExecutionProvider']
        self._using_cuda = "CUDAExecutionProvider" in self._providers
        self.session = self.ort.InferenceSession(model_path, providers=self._providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # Class names (defaults to COCO-ish stub, overridden by env/sidecars)
        self.names = _load_class_names(self.model_path, class_list, class_paths, GAME_ID)

    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self._imgsz / h, self._imgsz / w)
        nh, nw = int(h * scale), int(w * scale)
        if nh != h or nw != w:
            img = cv2.resize(img, (nw, nh))
        
        # Pad to square
        top = (self._imgsz - nh) // 2
        bottom = self._imgsz - nh - top
        left = (self._imgsz - nw) // 2
        right = self._imgsz - nw - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32') / 255.0
        return img[None], scale, top, left

    def detect(self, frame: np.ndarray) -> Iterable[Detection]:
        blob, scale, pad_h, pad_w = self._preprocess(frame)
        try:
            outputs = self.session.run(self.output_names, {self.input_name: blob})
        except Exception as exc:  # GPU OOM or provider failure
            if self._using_cuda:
                logger.warning("ONNX runtime failed on CUDA (%s), falling back to CPU", exc)
                self._providers = ['CPUExecutionProvider']
                self._using_cuda = False
                self.session = self.ort.InferenceSession(self.model_path, providers=self._providers)
                outputs = self.session.run(self.output_names, {self.input_name: blob})
            else:
                raise
        # Output shape [1, 84, 8400] -> [Batch, 4+classes, anchors]
        det = outputs[0][0]
        
        # Post-process (simplified NMS)
        # Transpose to [8400, 84]
        det = det.T
        boxes, scores, class_ids = [], [], []
        
        for i in range(det.shape[0]):
            row = det[i]
            # Class scores start at index 4
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            
            if score > self._conf:
                # Extract box (cx, cy, w, h)
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                # Unpad and unscale
                x1 = (cx - bw / 2 - pad_w) / scale
                y1 = (cy - bh / 2 - pad_h) / scale
                x2 = (cx + bw / 2 - pad_w) / scale
                y2 = (cy + bh / 2 - pad_h) / scale
                
                boxes.append([x1, y1, x2 - x1, y2 - y1]) # XYWH for NMS
                scores.append(float(score))
                class_ids.append(class_id)

        if not boxes: return []
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self._conf, 0.45)
        detections = []
        for idx in indices:
            if isinstance(idx, (int, float)):
                i = int(idx)
            elif isinstance(idx, np.ndarray):
                i = int(idx.item() if idx.ndim == 0 else idx.flatten()[0])
            elif isinstance(idx, (list, tuple)):
                i = int(idx[0])
            else:
                try:
                    i = int(idx)
                except Exception:
                    i = int(np.array(idx).flatten()[0])
            box = boxes[i]
            # XYWH to XYXY
            x, y, w, h = box
            detections.append(Detection(
                label=self.names.get(class_ids[i], str(class_ids[i])),
                confidence=scores[i],
                box=[x, y, x + w, y + h]
            ))
        return detections

    def update_runtime(self, config: dict):
        self._conf = float(config.get("conf", self._conf))


def build_detector(
    backend: str,
    model_path: str,
    device: str,
    conf: float,
    imgsz: int,
    class_list: str = "",
    class_paths: str = "",
) -> BaseDetector:
    if backend in {"ultralytics", "torch", "trt", "tensorrt"}:
        try:
            if backend in {"trt", "tensorrt"} and not model_path.endswith(".engine"):
                logger.warning(
                    "DETECTOR_BACKEND=%s selected but OBJECT_MODEL_PATH is not a .engine (%s). "
                    "This backend is an alias routed to UltralyticsDetector; provide a TensorRT .engine for true TRT path.",
                    backend,
                    model_path,
                )
            detector = UltralyticsDetector(model_path, device, conf, imgsz, class_list, class_paths)
            detector.backend = backend
            logger.info(
                "Detector init ok: backend=%s impl=UltralyticsDetector model=%s device=%s imgsz=%s",
                backend,
                model_path,
                device,
                imgsz,
            )
            return detector
        except Exception as exc:
            logger.error("Failed to init Ultralytics detector: %s", exc)
    elif backend == "onnx":
        try:
            detector = OnnxDetector(model_path, device, conf, imgsz, class_list, class_paths)
            detector.backend = backend
            return detector
        except Exception as exc:
            logger.error("Failed to init ONNX detector: %s", exc)
    else:
        logger.info("Object detector backend %s not recognized, using dummy", backend)
    return DummyDetector()


class ObjectDetectionAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="object_detection_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self.detector = None
        self.fallback_detector = None
        self.detector_lock = threading.Lock()
        self.vision_mode = VISION_MODE_DEFAULT if VISION_MODE_DEFAULT in MODE_SETTINGS else "medium"
        settings = MODE_SETTINGS.get(self.vision_mode, MODE_SETTINGS["medium"])
        self.frame_stride = settings.get("frame_stride", 1)
        self._frame_counter = 0
        self._apply_detector_settings(settings)
        
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        self.worker_thread.start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self.worker_thread.join(timeout=2)
        logger.info("Object detection agent shut down.")

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            topics = [(FRAME_TOPIC, 0)]
            if VISION_CONFIG_TOPIC:
                topics.append((VISION_CONFIG_TOPIC, 0))
            client.subscribe(topics)
            logger.info("Subscribed to %s", [t for t, _ in topics])
            client.publish(OBJECT_TOPIC, json.dumps({"ok": True, "event": "object_detector_ready"}))
        else:
            logger.error("MQTT connect failed: rc=%s", _as_int(rc))
            client.publish(OBJECT_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0:
            logger.warning("MQTT disconnected: rc=%s", _as_int(rc))

    def _on_message(self, client, _userdata, msg):
        if msg.topic == VISION_CONFIG_TOPIC:
            self._handle_config(msg.payload)
            return

        self._frame_counter += 1
        if self.frame_stride > 1 and self._frame_counter % self.frame_stride != 0:
            return

        try:
            self.frame_queue.put_nowait(msg.payload)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.frame_queue.put_nowait(msg.payload)
            except queue.Full:
                pass
            logger.debug("Frame queue full; dropped oldest frame.")

    def _worker_loop(self):
        """Worker thread to process frames from the queue."""
        while not stop_event.is_set():
            try:
                payload_bytes = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                payload = json.loads(payload_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug("Non-JSON frame payload dropped in worker")
                continue

            frame_bytes = get_frame_bytes(payload)
            frame = decode_frame(frame_bytes)
            if frame is None:
                continue

            self._ensure_detector()
            detect_start = time.perf_counter()
            detections = []
            used_detector = self.detector
            fallback_used = False
            try:
                detections = self.detector.detect(frame)
            except Exception:  # noqa: BLE001
                logger.exception("Primary detector failed")
                detections = []
            if not detections and self.fallback_detector is not None:
                try:
                    detections = self.fallback_detector.detect(frame)
                    used_detector = self.fallback_detector
                    fallback_used = True
                except Exception:  # noqa: BLE001
                    logger.exception("Fallback detector failed")
                    detections = []
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            emit_latency(
                self.client,
                "detect",
                detect_ms,
                sla_ms=SLA_STAGE_DETECT_MS,
                tags={"frame_ts": payload.get("timestamp"), "fallback": str(fallback_used)},
                agent="object_detection_agent",
            )
            logger.info("Detected %d objects", len(detections))
            result_payload = {
                "ok": True,
                "timestamp": time.time(),
                "frame_ts": payload.get("timestamp", time.time()),
                "objects": [det.as_dict() for det in detections],
                "backend": getattr(used_detector, "backend", DETECTOR_BACKEND),
                "fallback_used": fallback_used,
            }
            self.client.publish(OBJECT_TOPIC, json.dumps(result_payload))

    def _ensure_detector(self):
        if self.detector is not None:
            return
        with self.detector_lock:
            if self.detector is not None:
                return
            try:
                self.detector = build_detector(
                    DETECTOR_BACKEND,
                    MODEL_PATH,
                    DEVICE,
                    CONF_THRESHOLD,
                    IMG_SIZE,
                    CLASS_LIST,
                    CLASS_PATHS,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Detector init failed, falling back to dummy: %s", exc)
                self.detector = DummyDetector()
            logger.info("Detector initialized: backend=%s impl=%s", DETECTOR_BACKEND, type(self.detector).__name__)
            if hasattr(self.detector, "update_runtime"):
                settings = getattr(self, "_pending_settings", MODE_SETTINGS.get(self.vision_mode, MODE_SETTINGS["medium"]))
                self.detector.update_runtime(settings)
            if FALLBACK_BACKEND and FALLBACK_MODEL_PATH:
                device = FALLBACK_DEVICE or DEVICE
                try:
                    self.fallback_detector = build_detector(
                        FALLBACK_BACKEND,
                        FALLBACK_MODEL_PATH,
                        device,
                        FALLBACK_CONF_THRESHOLD,
                        FALLBACK_IMG_SIZE,
                        FALLBACK_CLASS_LIST,
                        FALLBACK_CLASS_PATHS,
                    )
                    logger.info(
                        "Fallback detector initialized: backend=%s impl=%s",
                        FALLBACK_BACKEND,
                        type(self.fallback_detector).__name__,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Fallback detector init failed: %s", exc)
                    self.fallback_detector = None

    def _apply_detector_settings(self, settings: dict):
        self._pending_settings = settings
        if self.detector is not None and hasattr(self.detector, "update_runtime"):
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


def _handle_signal(signum, frame):
    logger.info("Signal received, shutting down.")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = ObjectDetectionAgent()
    agent.start()


if __name__ == "__main__":
    main()
