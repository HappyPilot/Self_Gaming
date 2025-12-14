"""Heuristic player locator that tags likely hero coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from core.observations import DetectedObject


@dataclass
class PlayerLocatorConfig:
    center_x: float = 0.5
    center_y: float = 0.55
    max_offset: float = 0.45
    min_area: float = 0.01
    roi_box: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.9)
    min_roi_area: float = 0.02
    min_det_score: float = 0.02
    label: str = "player"


class PlayerLocator:
    """Selects a player candidate based on detections or ROI saliency."""

    def __init__(self, config: PlayerLocatorConfig | None = None) -> None:
        self.cfg = config or PlayerLocatorConfig()

    def locate(
        self,
        frame: np.ndarray,
        detections: Iterable[DetectedObject],
    ) -> Optional[DetectedObject]:
        candidate = self._from_detections(detections)
        if candidate is not None:
            return candidate
        return self._from_frame(frame)

    # ----------------------------------------------------------------- helpers
    def _score_detection(self, det: DetectedObject) -> float:
        x1, y1, x2, y2 = det.bbox
        area = max(1e-5, (x2 - x1) * (y2 - y1))
        if area < self.cfg.min_area:
            return 0.0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = cx - self.cfg.center_x
        dy = cy - self.cfg.center_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.cfg.max_offset:
            return 0.0
        proximity = max(0.0, 1.0 - (dist / max(1e-5, self.cfg.max_offset)))
        confidence = max(0.05, float(det.confidence or 0.0))
        return area * proximity * confidence

    def _from_detections(self, detections: Iterable[DetectedObject]) -> Optional[DetectedObject]:
        best_det = None
        best_score = self.cfg.min_det_score
        for det in detections:
            score = self._score_detection(det)
            if score > best_score:
                best_det = det
                best_score = score
        if best_det is None:
            return None
        return DetectedObject(
            label=self.cfg.label,
            confidence=float(best_det.confidence or 0.0),
            bbox=best_det.bbox,
            extra={"source": "yolo"},
        )

    def _from_frame(self, frame: np.ndarray) -> Optional[DetectedObject]:
        if frame is None or frame.size == 0:
            return None
        y1f, x1f, y2f, x2f = self._roi_pixels(frame)
        roi = frame[y1f:y2f, x1f:x2f]
        if roi.size == 0:
            return None
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        h, w = frame.shape[:2]
        roi_area = (x2f - x1f) * (y2f - y1f)
        for contour in contours:
            if cv2.contourArea(contour) < 8:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            area = (cw * ch) / max(1.0, float(roi_area))
            if area < self.cfg.min_roi_area:
                continue
            nx1 = (x + x1f) / float(w)
            ny1 = (y + y1f) / float(h)
            nx2 = (x + x1f + cw) / float(w)
            ny2 = (y + y1f + ch) / float(h)
            return DetectedObject(
                label=self.cfg.label,
                confidence=0.35,
                bbox=(nx1, ny1, nx2, ny2),
                extra={"source": "roi"},
            )
        return None

    def _roi_pixels(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1 = int(max(0.0, min(1.0, self.cfg.roi_box[0])) * w)
        y1 = int(max(0.0, min(1.0, self.cfg.roi_box[1])) * h)
        x2 = int(max(0.0, min(1.0, self.cfg.roi_box[2])) * w)
        y2 = int(max(0.0, min(1.0, self.cfg.roi_box[3])) * h)
        if x2 <= x1:
            x2 = min(w, x1 + int(0.1 * w))
        if y2 <= y1:
            y2 = min(h, y1 + int(0.1 * h))
        return y1, x1, y2, x2
