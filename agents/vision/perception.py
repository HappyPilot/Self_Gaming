"""Composable perception pipeline combining object detection and OCR."""
from __future__ import annotations

import abc
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from core.observations import DetectedObject, GameObservation, OcrZone
from vision.player_locator import PlayerLocator
from vision.regions import RegionBox, TEXT_REGIONS, crop_region, normalize_box


class ObjectDetectorBackend(abc.ABC):
    """Base interface for YOLO / detection engines."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Iterable[DetectedObject]:
        """Return detected objects for the frame."""


class OcrBackend(abc.ABC):
    """Base interface for OCR engines."""

    @abc.abstractmethod
    def recognize_region(
        self,
        region_name: str,
        region_frame: np.ndarray,
        bbox: RegionBox,
    ) -> Optional[OcrZone]:
        """Return OCR info for the region (or None if nothing meaningful)."""


class PerceptionPipeline:
    """Glue code that produces GameObservation objects."""

    def __init__(
        self,
        detector: ObjectDetectorBackend,
        ocr: OcrBackend,
        text_regions: Mapping[str, RegionBox] | None = None,
        player_locator: PlayerLocator | None = None,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.player_locator = player_locator
        if text_regions:
            self.text_regions = {name: normalize_box(box) for name, box in text_regions.items()}
        else:
            self.text_regions = dict(TEXT_REGIONS)

    def build_observation(self, frame: np.ndarray, frame_id: int | None = None) -> GameObservation:
        fid = frame_id if frame_id is not None else -1
        detections = list(self.detector.detect(frame, frame_id=fid))
        zones: Dict[str, OcrZone] = {}
        player = None
        if self.player_locator is not None:
            player = self.player_locator.locate(frame, detections)
        objects_out = list(detections)
        if player is not None:
            objects_out.append(player)
        for region_name, box in self.text_regions.items():
            try:
                crop = crop_region(frame, box)
            except ValueError:
                continue
            zone = self.ocr.recognize_region(region_name, crop, box)
            if zone is not None:
                zones[region_name] = zone
        return GameObservation(
            frame_id=fid,
            yolo_objects=objects_out,
            text_zones=zones,
            player_candidate=player,
        )
