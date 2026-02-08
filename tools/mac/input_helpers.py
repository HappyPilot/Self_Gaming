"""Helpers for resolving input targets."""
from __future__ import annotations

from typing import Optional, Tuple


def clamp_point(x: int, y: int, bounds: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    if not bounds:
        return x, y
    left, top, right, bottom = bounds
    return max(left, min(int(x), right)), max(top, min(int(y), bottom))


def resolve_target_point(data: dict, bounds: Optional[Tuple[int, int, int, int]], screen_size: Tuple[int, int]):
    if not isinstance(data, dict):
        return None
    target_px = data.get("target_px")
    if isinstance(target_px, (list, tuple)) and len(target_px) == 2:
        try:
            x = int(round(float(target_px[0])))
            y = int(round(float(target_px[1])))
        except (TypeError, ValueError):
            return None
        return clamp_point(x, y, bounds)
    target_norm = data.get("target_norm")
    if not isinstance(target_norm, (list, tuple)) or len(target_norm) != 2:
        return None
    try:
        x_norm = float(target_norm[0])
        y_norm = float(target_norm[1])
    except (TypeError, ValueError):
        return None
    if bounds:
        left, top, right, bottom = bounds
        width = max(1, right - left)
        height = max(1, bottom - top)
        x = left + x_norm * width
        y = top + y_norm * height
    else:
        width, height = screen_size
        x = x_norm * width
        y = y_norm * height
    return clamp_point(int(round(x)), int(round(y)), bounds)
