"""EasyOCR-backed implementation of the OcrBackend interface."""
from __future__ import annotations

import importlib
from typing import Iterable, Optional, Sequence

import numpy as np
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from core.observations import OcrZone
from vision.perception import OcrBackend
from vision.regions import RegionBox


def _import_easyocr():
    try:
        return importlib.import_module("easyocr")
    except ImportError as exc:  # pragma: no cover - exercised via tests with missing dependency
        raise RuntimeError("easyocr is not installed. Install it to enable EasyOCR backend.") from exc


class EasyOcrBackend(OcrBackend):
    """Wraps easyocr.Reader to work with the perception pipeline."""

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        gpu: bool = True,
        reader_kwargs: Optional[dict] = None,
        min_confidence: float = 0.05,
    ) -> None:
        easyocr = _import_easyocr()
        lang_list: Iterable[str] = [lang.strip() for lang in (languages or ["en"])]
        kwargs = dict(reader_kwargs or {})
        kwargs.setdefault("gpu", gpu)
        kwargs.setdefault("verbose", False)
        self.reader = easyocr.Reader(list(lang_list), **kwargs)
        self.min_confidence = min_confidence

    def recognize_region(
        self,
        region_name: str,
        region_frame: np.ndarray,
        bbox: RegionBox,
    ) -> Optional[OcrZone]:
        if region_frame is None or region_frame.size == 0:
            return None
        if torch is not None:
            with torch.no_grad():
                results = self.reader.readtext(region_frame)
        else:
            results = self.reader.readtext(region_frame)
        if not results:
            return None
        best_text = ""
        best_conf = self.min_confidence
        for entry in results:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                text = str(entry[1]).strip()
                try:
                    conf = float(entry[2])
                except (TypeError, ValueError):
                    continue
                if text and conf >= best_conf:
                    best_text = text
                    best_conf = conf
        if not best_text:
            return None
        return OcrZone(region_name=region_name, text=best_text, confidence=float(best_conf), bbox=bbox)
