"""TensorRT-backed YOLO11 detector backend."""
from __future__ import annotations

from typing import Iterable, Optional
from ultralytics import YOLO

from core.observations import DetectedObject
from vision.perception import ObjectDetectorBackend
from vision.yolo11_torch_backend import _result_to_objects


class Yolo11TensorRTBackend(ObjectDetectorBackend):
    """Run YOLO11 TensorRT engines exported via Ultralytics or trtexec."""

    def __init__(self, engine_path: str, conf: float = 0.35, imgsz: int = 640):
        if not engine_path:
            raise ValueError("engine_path is required for TensorRT backend")
        self.model = YOLO(engine_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame, frame_id: Optional[int] = None) -> Iterable[DetectedObject]:
        if frame is None or frame.size == 0:
            return []
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        return _result_to_objects(results[0], frame.shape)
