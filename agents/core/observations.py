"""Shared dataclasses describing what agents perceive on screen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BoundingBox = Tuple[float, float, float, float]


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: BoundingBox
    extra: Optional[dict] = None


@dataclass
class OcrZone:
    region_name: str
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None


@dataclass
class GameObservation:
    frame_id: int
    yolo_objects: List[DetectedObject] = field(default_factory=list)
    text_zones: Dict[str, OcrZone] = field(default_factory=dict)
    player_candidate: Optional[DetectedObject] = None
