"""RapidOCR (Paddle ONNX) backend implementation."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from core.observations import OcrZone
from vision.perception import OcrBackend
from vision.regions import RegionBox


def _import_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        return RapidOCR
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "rapidocr_onnxruntime is not installed. Install it to enable the RapidOCR backend."
        ) from exc


class RapidOcrBackend(OcrBackend):
    """ONNXRuntime-based OCR backend using RapidOCR pipelines."""

    def __init__(
        self,
        *,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
        cls_model_path: str | None = None,
        providers: Iterable[str] | None = None,
        min_confidence: float = 0.05,
    ) -> None:
        RapidOCR = _import_rapidocr()
        kwargs = {
            "det_model_path": det_model_path,
            "rec_model_path": rec_model_path,
            "cls_model_path": cls_model_path,
            "providers": list(providers) if providers else None,
        }
        # Remove None entries to let RapidOCR choose defaults/downloads.
        kwargs = {k: v for k, v in kwargs.items() if v}
        self.ocr = RapidOCR(**kwargs)
        self.min_confidence = float(min_confidence)

    def recognize_region(
        self,
        region_name: str,
        region_frame: np.ndarray,
        bbox: RegionBox,
    ) -> Optional[OcrZone]:
        if region_frame is None or region_frame.size == 0:
            return None
        try:
            results, _ = self.ocr(region_frame)
        except Exception:
            return None
        if not results:
            return None
        best_text = ""
        best_conf = self.min_confidence
        for candidate in results:
            if not candidate or len(candidate) < 3:
                continue
            text = str(candidate[1]).strip()
            try:
                score = float(candidate[2])
            except (TypeError, ValueError):
                continue
            if text and score >= best_conf:
                best_text = text
                best_conf = score
        if not best_text:
            return None
        return OcrZone(region_name=region_name, text=best_text, confidence=float(best_conf), bbox=bbox)
