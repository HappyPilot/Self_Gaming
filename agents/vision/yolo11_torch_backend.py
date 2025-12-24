"""Ultralytics YOLO11 backend that returns DetectedObject list."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List, Optional

import numpy as np
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from core.observations import DetectedObject
from vision.perception import ObjectDetectorBackend


def _result_to_objects(result, frame_shape) -> List[DetectedObject]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    frame_h, frame_w = frame_shape[:2]
    names = result.names or {}
    detected: List[DetectedObject] = []
    for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        x1, y1, x2, y2 = xyxy
        nx1 = max(0.0, min(1.0, x1 / frame_w))
        ny1 = max(0.0, min(1.0, y1 / frame_h))
        nx2 = max(0.0, min(1.0, x2 / frame_w))
        ny2 = max(0.0, min(1.0, y2 / frame_h))
        cls_idx = int(cls)
        label = names.get(cls_idx, f"cls_{cls_idx}") if isinstance(names, dict) else str(cls_idx)
        detected.append(
            DetectedObject(
                label=label,
                confidence=float(conf),
                bbox=(nx1, ny1, nx2, ny2),
            )
        )
    return detected


class Yolo11TorchBackend(ObjectDetectorBackend):
    """Torch-based YOLO11 detector that outputs normalized boxes."""

    def __init__(self, weights_path: str, device: str = "cuda:0", conf: float = 0.35, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ultralytics is not installed. Install it to use yolo11_torch backend.") from exc
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Iterable[DetectedObject]:
        if frame is None or frame.size == 0:
            return []
        ctx = torch.no_grad() if torch is not None else nullcontext()
        with ctx:
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
            )
        if not results:
            return []
        return _result_to_objects(results[0], frame.shape)
