"""Ultralytics YOLO-World backend for open-vocabulary detection."""
from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Iterable, List, Optional

import cv2
import numpy as np
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from core.observations import DetectedObject
from vision.perception import ObjectDetectorBackend

logger = logging.getLogger(__name__)


@contextmanager
def _clip_on_cpu():
    """Temporarily force CLIP text encoder to CPU to avoid GPU allocator crashes."""
    if torch is None:
        yield
        return
    orig_is_available = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False  # type: ignore
        yield
    finally:
        torch.cuda.is_available = orig_is_available


class YoloWorldBackend(ObjectDetectorBackend):
    """Torch-based YOLO-World detector that supports dynamic text prompts."""

    def __init__(
        self,
        weights_path: str,
        classes: list[str],
        device: str = "cuda:0",
        conf: float = 0.25,
        imgsz: int = 640,
        max_size: int | None = None,
        fallback_cpu: bool = False,
        clip_cpu: bool = True,
    ):
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ultralytics is not installed. Install it to use yolo_world backend.") from exc
        self.model = YOLOWorld(weights_path)
        try:
            if clip_cpu:
                with _clip_on_cpu():
                    self.model.set_classes(classes)
            else:
                self.model.set_classes(classes)
        except Exception:  # noqa: BLE001
            logger.exception("YOLO-World set_classes failed; continuing without classes override")
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.max_size = max_size if max_size and max_size > 0 else None
        self.fallback_cpu = fallback_cpu
        logger.info(
            "YOLO-World backend ready weights=%s device=%s conf=%.3f imgsz=%s classes=%s max_size=%s fallback_cpu=%s clip_cpu=%s",
            weights_path,
            device,
            conf,
            imgsz,
            classes,
            self.max_size,
            self.fallback_cpu,
            clip_cpu,
        )

    def _predict(self, frame: np.ndarray, device: str):
        ctx = torch.no_grad() if torch is not None else nullcontext()
        with ctx:
            use_half = device.startswith("cuda")
            return self.model.predict(
                source=frame,
                device=device,
                conf=self.conf,
                imgsz=self.imgsz,
                half=use_half,
                verbose=False,
            )

    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Iterable[DetectedObject]:
        if frame is None or frame.size == 0:
            return []
        try:
            infer_frame = frame
            scale = 1.0
            if self.max_size:
                h, w = frame.shape[:2]
                longer = max(h, w)
                if longer > self.max_size:
                    scale = float(self.max_size) / float(longer)
                    infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            results = self._predict(infer_frame, self.device)
        except Exception as exc:  # noqa: BLE001
            logger.exception("YOLO-World inference failed on device=%s", self.device)
            if not self.fallback_cpu or self.device.startswith("cpu"):
                return []
            try:
                logger.info("Retrying YOLO-World on CPU due to failure")
                results = self._predict(infer_frame, "cpu")
            except Exception:  # noqa: BLE001
                logger.exception("YOLO-World CPU fallback failed")
                return []

        if not results:
            return []
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []
        frame_h, frame_w = frame.shape[:2]
        detected: List[DetectedObject] = []
        if self.classes:
            names = {i: name for i, name in enumerate(self.classes)}
        else:
            names = res.names or {}
        raw_boxes = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        if scores.size:
            logger.debug("YOLO-World detections=%d max_score=%.4f frame_id=%s", scores.size, float(scores.max()), frame_id)
        for (x1, y1, x2, y2), conf, cls in zip(raw_boxes, scores, classes):
            if scale != 1.0:
                x1, y1, x2, y2 = [coord / scale for coord in (x1, y1, x2, y2)]
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
