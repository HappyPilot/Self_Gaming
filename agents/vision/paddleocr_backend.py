"""PaddleOCR-backed implementation of the OcrBackend interface."""
from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np

from core.observations import OcrZone
from vision.perception import OcrBackend
from vision.regions import RegionBox


def _import_paddleocr():
    try:
        return importlib.import_module("paddleocr")
    except ImportError as exc:  # pragma: no cover - handled via runtime guard
        raise RuntimeError("paddleocr is not installed. Install PaddleOCR to enable this backend.") from exc


class PaddleOcrBackend(OcrBackend):
    """Use PaddleOCR's Python API for HUD text recognition."""

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        paddle_kwargs: Optional[dict] = None,
        min_confidence: float = 0.05,
    ) -> None:
        paddleocr = _import_paddleocr()
        kwargs: dict[str, Any] = dict(paddle_kwargs or {})
        kwargs.setdefault("lang", lang)
        kwargs.setdefault("use_angle_cls", use_angle_cls)
        kwargs.setdefault("show_log", False)
        self.ocr = paddleocr.PaddleOCR(**kwargs)
        self.min_confidence = min_confidence

    def recognize_region(
        self,
        region_name: str,
        region_frame: np.ndarray,
        bbox: RegionBox,
    ) -> Optional[OcrZone]:
        if region_frame is None or region_frame.size == 0:
            return None
        try:
            results = self.ocr.ocr(region_frame, cls=True)
        except Exception:
            return None
        if not results:
            return None
        best_text = ""
        best_conf = self.min_confidence
        for result in results:
            if not result or len(result) < 2:
                continue
            text_block = result[1]
            if isinstance(text_block, (list, tuple)) and len(text_block) >= 2:
                text = str(text_block[0]).strip()
                try:
                    conf = float(text_block[1])
                except (TypeError, ValueError):
                    continue
                if text and conf >= best_conf:
                    best_text = text
                    best_conf = conf
        if not best_text:
            return None
        return OcrZone(region_name=region_name, text=best_text, confidence=float(best_conf), bbox=bbox)
