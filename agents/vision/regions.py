"""Screen region utilities for OCR-aware perception."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Tuple

import numpy as np

RegionBox = Tuple[float, float, float, float]


# Default layout tuned for 16:9 games (Path of Exile, etc.).
TEXT_REGIONS: Dict[str, RegionBox] = {
    "quest_log": (0.72, 0.05, 0.98, 0.30),
    "chat_log": (0.00, 0.70, 0.35, 0.98),
    "top_center_info": (0.30, 0.00, 0.70, 0.08),
    "tooltip_area": (0.50, 0.35, 0.95, 0.75),
    "death_dialog": (0.42, 0.58, 0.58, 0.76),
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_box(box: RegionBox) -> RegionBox:
    """Ensure the box has sane bounds within [0, 1]."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((_clamp(x1), _clamp(x2)))
    y1, y2 = sorted((_clamp(y1), _clamp(y2)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError(f"Invalid region bounds: {box}")
    return x1, y1, x2, y2


def crop_region(frame: np.ndarray, region_box: RegionBox) -> np.ndarray:
    """Return a cropped sub-frame for the normalized region box."""
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim < 2:
        raise ValueError("frame must have at least 2 dimensions")

    box = normalize_box(region_box)
    height, width = frame.shape[:2]
    x1 = int(round(box[0] * width))
    y1 = int(round(box[1] * height))
    x2 = int(round(box[2] * width))
    y2 = int(round(box[3] * height))
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return frame[y1:y2, x1:x2].copy()


def iter_regions(
    frame: np.ndarray,
    regions: Mapping[str, RegionBox] | None = None,
) -> Iterator[Tuple[str, RegionBox, np.ndarray]]:
    """Yield (name, normalized_box, crop) for each configured region."""
    regions = regions or TEXT_REGIONS
    for name, box in regions.items():
        try:
            normalized = normalize_box(box)
        except ValueError:
            continue
        yield name, normalized, crop_region(frame, normalized)
