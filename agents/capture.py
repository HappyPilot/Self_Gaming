import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from capture_gstreamer import build_gstreamer_pipeline

logger = logging.getLogger("capture")


@dataclass
class FrameHandle:
    frame: np.ndarray
    timestamp: float
    source: str = "unknown"

    def to_numpy(self) -> np.ndarray:
        return self.frame


class CaptureAdapter:
    name = "base"

    def start(self) -> bool:
        raise NotImplementedError

    def read(self) -> Optional[FrameHandle]:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def describe(self) -> str:
        return self.name


class DummyCapture(CaptureAdapter):
    name = "dummy"

    def __init__(self) -> None:
        width = int(os.getenv("VIDEO_WIDTH", "640")) or 640
        height = int(os.getenv("VIDEO_HEIGHT", "360")) or 360
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def start(self) -> bool:
        return True

    def read(self) -> Optional[FrameHandle]:
        return FrameHandle(self._frame.copy(), time.time(), source="dummy")


class OpenCvCapture(CaptureAdapter):
    name = "v4l2"

    def __init__(self) -> None:
        self._cap = None
        self._active_device = None
        self._device = os.getenv("VIDEO_DEVICE", "/dev/video0")
        self._fallbacks = [
            p.strip()
            for p in os.getenv("VIDEO_DEVICE_FALLBACKS", "/dev/video2,/dev/video1,0,2").split(",")
            if p.strip()
        ]
        self._width = int(os.getenv("VIDEO_WIDTH", "0"))
        self._height = int(os.getenv("VIDEO_HEIGHT", "0"))
        self._fps = float(os.getenv("VIDEO_FPS", "0"))
        self._pixfmt = os.getenv("VIDEO_PIXFMT", "")[:4]

    def start(self) -> bool:
        candidates = [self._device] + self._fallbacks
        for cand_str in candidates:
            cand = int(cand_str) if cand_str.isdigit() else cand_str
            cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
            if cap.isOpened():
                self._configure_capture(cap)
                self._cap = cap
                self._active_device = cand
                return True
            cap.release()
        return False

    def _configure_capture(self, cap) -> None:
        if self._width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        if self._pixfmt:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._pixfmt))
            except Exception:
                pass

    def read(self) -> Optional[FrameHandle]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return FrameHandle(frame, time.time(), source=str(self._active_device))

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"v4l2 device={self._active_device}"


class GStreamerCapture(CaptureAdapter):
    name = "gstreamer"

    def __init__(self) -> None:
        self._cap = None
        device = os.getenv("VIDEO_DEVICE", "/dev/video0")
        width = int(os.getenv("VIDEO_WIDTH", "0"))
        height = int(os.getenv("VIDEO_HEIGHT", "0"))
        fps = float(os.getenv("VIDEO_FPS", "0"))
        use_nvmm = os.getenv("GST_USE_NVMM", "0") not in {"0", "false", "no"}
        self._pipeline = build_gstreamer_pipeline(device, width, height, fps, use_nvmm)

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            logger.warning("Failed to open GStreamer pipeline: %s", self._pipeline)
            self._cap.release()
            self._cap = None
            return False
        return True

    def read(self) -> Optional[FrameHandle]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return FrameHandle(frame, time.time(), source="gstreamer")

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"pipeline={self._pipeline}"


class MssCapture(CaptureAdapter):
    name = "mss"

    def __init__(self) -> None:
        self._ctx = None
        self._monitor = None
        self._mss = None
        try:
            import mss  # type: ignore
        except Exception:
            mss = None
        self._mss = mss

    def start(self) -> bool:
        if self._mss is None:
            logger.warning("MSS backend requested but mss is not installed")
            return False
        self._ctx = self._mss.mss()
        monitor_idx = int(os.getenv("MSS_MONITOR", "1"))
        monitors = self._ctx.monitors
        if monitor_idx < 0 or monitor_idx >= len(monitors):
            monitor_idx = 1
        self._monitor = monitors[monitor_idx]
        return True

    def read(self) -> Optional[FrameHandle]:
        if self._ctx is None or self._monitor is None:
            return None
        shot = self._ctx.grab(self._monitor)
        frame = np.array(shot)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return FrameHandle(frame, time.time(), source="mss")

    def stop(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
            self._monitor = None


def build_capture_backend(kind: str | None) -> CaptureAdapter:
    normalized = (kind or os.getenv("CAPTURE_BACKEND", "v4l2")).strip().lower()
    if normalized in {"gstreamer", "gst"}:
        return GStreamerCapture()
    if normalized in {"mss", "screen"}:
        return MssCapture()
    if normalized in {"dummy", "none"}:
        return DummyCapture()
    return OpenCvCapture()
