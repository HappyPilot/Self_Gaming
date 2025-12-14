"""Pure TensorRT engine backend for YOLO11 (no PyTorch/Ultralytics)."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

from core.observations import DetectedObject
from vision.perception import ObjectDetectorBackend

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    """Resize with stretch (no padding) for simplicity."""
    h, w = image.shape[:2]
    scale_w = size / float(w)
    scale_h = size / float(h)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, scale_w, scale_h


class YoloTRTEngineBackend(ObjectDetectorBackend):
    """Minimal TRT inference for YOLO11 exported engine."""

    def __init__(
        self,
        engine_path: str,
        conf: float = 0.25,
        imgsz: int = 416,
        class_names: Optional[List[str]] = None,
    ):
        if not engine_path:
            raise ValueError("engine_path is required")
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names or []

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
        self.context = self.engine.create_execution_context()
        # Bindings
        self.input_idx = 0
        self.output_idx = 1 if self.engine.num_bindings > 1 else 0
        # Set shape for dynamic input if needed
        try:
            self.context.set_binding_shape(self.input_idx, (1, 3, imgsz, imgsz))
        except Exception:
            pass

        # Allocate buffers
        input_shape = self.engine.get_binding_shape(self.input_idx)
        output_shape = self.engine.get_binding_shape(self.output_idx)
        self.output_elements = int(np.prod(output_shape))

        self.d_input = cuda.mem_alloc(trt.volume(input_shape) * np.dtype(np.float32).itemsize)
        self.h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        logger.info(
            "TRT engine loaded: %s input_shape=%s output_shape=%s conf=%.3f",
            engine_path,
            tuple(input_shape),
            tuple(output_shape),
            conf,
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        resized, scale_w, scale_h = _letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        chw = np.transpose(tensor, (2, 0, 1))[None, ...].copy(order="C")
        return chw, scale_w, scale_h

    def _infer(self, inp: np.ndarray) -> np.ndarray:
        cuda.memcpy_htod_async(self.d_input, inp, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return np.array(self.h_output, copy=True)

    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Iterable[DetectedObject]:
        if frame is None or frame.size == 0:
            return []
        try:
            inp, scale_w, scale_h = self._preprocess(frame)
            raw = self._infer(inp)
        except Exception:  # noqa: BLE001
            logger.exception("TRT inference failed")
            return []

        # Output reshape: assume (1, C, N)
        try:
            output_shape = self.engine.get_binding_shape(self.output_idx)
            c, n = output_shape[1], output_shape[2]
            preds = raw.reshape((1, c, n))
        except Exception:
            logger.exception("Failed to reshape output")
            return []

        preds = preds[0]  # C x N
        boxes_xywh = preds[0:4, :].T  # N x 4
        obj_scores = _sigmoid(preds[4, :])
        cls_scores = _sigmoid(preds[5:, :].T)  # N x num_classes

        detected: List[DetectedObject] = []
        frame_h, frame_w = frame.shape[:2]

        for i in range(boxes_xywh.shape[0]):
            if cls_scores.shape[1] > 0:
                cls_idx = int(np.argmax(cls_scores[i]))
                cls_conf = cls_scores[i, cls_idx]
            else:
                cls_idx, cls_conf = 0, 1.0
            conf = float(obj_scores[i] * cls_conf)
            if conf < self.conf:
                continue
            xc, yc, w, h = boxes_xywh[i]
            # Map back to original frame coordinates
            xc_orig = xc / scale_w
            yc_orig = yc / scale_h
            w_orig = w / scale_w
            h_orig = h / scale_h
            x1 = max(0.0, xc_orig - w_orig / 2)
            y1 = max(0.0, yc_orig - h_orig / 2)
            x2 = min(frame_w - 1.0, xc_orig + w_orig / 2)
            y2 = min(frame_h - 1.0, yc_orig + h_orig / 2)
            nx1 = x1 / frame_w
            ny1 = y1 / frame_h
            nx2 = x2 / frame_w
            ny2 = y2 / frame_h
            label = (
                self.class_names[cls_idx]
                if self.class_names and 0 <= cls_idx < len(self.class_names)
                else f"cls_{cls_idx}"
            )
            detected.append(
                DetectedObject(
                    label=label,
                    confidence=conf,
                    bbox=(nx1, ny1, nx2, ny2),
                )
            )
        return detected
