"""Factories for detector backends used by perception and standalone agents."""
from __future__ import annotations

from typing import Any

from vision.perception import ObjectDetectorBackend


def build_detector_backend(
    kind: str | None,
    *,
    weights_path: str,
    device: str = "cuda:0",
    conf: float = 0.35,
    imgsz: int = 640,
    **kwargs: Any,
) -> ObjectDetectorBackend:
    """Return a detector backend configured per the supplied name."""

    if not weights_path:
        raise ValueError("weights_path must be provided for detector backend")
    normalized = (kind or "yolo11_torch").strip().lower()
    if normalized in {"yolo11", "yolo11_torch", "torch"}:
        from vision.yolo11_torch_backend import Yolo11TorchBackend

        return Yolo11TorchBackend(
            weights_path=weights_path,
            device=device,
            conf=conf,
            imgsz=imgsz,
        )
    if normalized in {"yolo11_trt", "tensorrt", "trt"}:
        from vision.yolo11_trt_backend import Yolo11TensorRTBackend

        engine_path = kwargs.get("engine_path", weights_path)
        return Yolo11TensorRTBackend(
            engine_path=engine_path,
            conf=conf,
            imgsz=imgsz,
        )
    if normalized in {"yoloworld", "yolo_world", "yolo-world"}:
        from vision.yolo_world_backend import YoloWorldBackend

        classes = kwargs.get("classes", []) or []
        max_size = kwargs.get("max_size")
        fallback_cpu = kwargs.get("fallback_cpu", False)
        clip_cpu = kwargs.get("clip_cpu", True)
        return YoloWorldBackend(
            weights_path=weights_path,
            classes=classes,
            device=device,
            conf=conf,
            imgsz=imgsz,
            max_size=max_size,
            fallback_cpu=fallback_cpu,
            clip_cpu=clip_cpu,
        )
    if normalized in {"yolo_trt_engine", "trt_engine", "trt"}:
        from vision.yolo_trt_engine_backend import YoloTRTEngineBackend

        class_names = kwargs.get("class_names")
        return YoloTRTEngineBackend(
            engine_path=weights_path,
            conf=conf,
            imgsz=imgsz,
            class_names=class_names,
        )
    raise ValueError(f"Unknown detector backend: {kind}")
