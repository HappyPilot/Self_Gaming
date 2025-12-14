"""OCR backend helpers and factory functions."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.observations import OcrZone
from vision.perception import OcrBackend
from vision.regions import RegionBox


class NullOcrBackend(OcrBackend):
    """Placeholder backend that never emits text."""

    def recognize_region(
        self,
        region_name: str,
        region_frame: np.ndarray,
        bbox: RegionBox,
    ) -> Optional[OcrZone]:
        return None


def build_ocr_backend(kind: str | None, **kwargs) -> OcrBackend:
    """Construct an OCR backend instance based on config name."""

    normalized = (kind or "null").strip().lower()
    if normalized in {"null", "none", "disabled"}:
        return NullOcrBackend()
    if normalized in {"easy", "easyocr"}:
        from vision.easyocr_backend import EasyOcrBackend

        languages: Sequence[str] | None = kwargs.get("languages")
        gpu = kwargs.get("gpu", True)
        reader_kwargs = kwargs.get("reader_kwargs")
        min_conf = kwargs.get("min_confidence", 0.05)
        return EasyOcrBackend(
            languages=languages,
            gpu=gpu,
            reader_kwargs=reader_kwargs,
            min_confidence=min_conf,
        )
    if normalized in {"paddle", "paddleocr"}:
        from vision.paddleocr_backend import PaddleOcrBackend

        lang = kwargs.get("lang")
        if not lang:
            languages: Sequence[str] | None = kwargs.get("languages")
            lang = languages[0] if languages else "en"
        min_conf = kwargs.get("min_confidence", 0.05)
        use_angle_cls = kwargs.get("use_angle_cls", True)
        paddle_kwargs = kwargs.get("paddle_kwargs")
        return PaddleOcrBackend(
            lang=lang,
            use_angle_cls=use_angle_cls,
            paddle_kwargs=paddle_kwargs,
            min_confidence=min_conf,
        )
    if normalized in {"rapidocr", "paddle_onnx", "paddle_rapid"}:
        from vision.rapidocr_backend import RapidOcrBackend

        min_conf = kwargs.get("min_confidence", 0.05)
        det_model_path = kwargs.get("det_model_path")
        rec_model_path = kwargs.get("rec_model_path")
        cls_model_path = kwargs.get("cls_model_path")
        providers = kwargs.get("providers")
        return RapidOcrBackend(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
            providers=providers,
            min_confidence=min_conf,
        )
    raise ValueError(f"Unknown OCR backend: {kind}")
