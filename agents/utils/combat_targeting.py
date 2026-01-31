"""Enemy targeting helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


def _center_from_bbox(bbox) -> Optional[Tuple[float, float]]:
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        return None
    try:
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
    except (TypeError, ValueError):
        return None
    if max(cx, cy) > 1.5:
        return None
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    return cx, cy


def _center_in_box(center: Tuple[float, float], box: Dict[str, float]) -> bool:
    try:
        x, y = float(center[0]), float(center[1])
        bx = float(box.get("x", 0.0))
        by = float(box.get("y", 0.0))
        bw = float(box.get("w", 0.0))
        bh = float(box.get("h", 0.0))
    except (TypeError, ValueError):
        return False
    return bx <= x <= (bx + bw) and by <= y <= (by + bh)


def pick_enemy_target(
    enemies: Iterable[dict],
    player_center: Tuple[float, float] = (0.5, 0.5),
    *,
    cluster_min: int = 3,
    hud_boxes: Optional[Iterable[Dict[str, float]]] = None,
    play_area: Optional[Dict[str, float]] = None,
) -> Optional[list[float]]:
    centers = []
    for obj in enemies or []:
        center = obj.get("center") or _center_from_bbox(obj.get("bbox") or obj.get("box"))
        if not center:
            continue
        cx, cy = center
        if hud_boxes:
            if any(_center_in_box((cx, cy), box) for box in hud_boxes):
                continue
        if play_area and not _center_in_box((cx, cy), play_area):
            continue
        centers.append((cx, cy))
    if not centers:
        return None
    cluster_min = max(2, int(cluster_min))
    if len(centers) >= cluster_min:
        avg_x = sum(c[0] for c in centers) / len(centers)
        avg_y = sum(c[1] for c in centers) / len(centers)
        return [avg_x, avg_y]
    px, py = player_center
    best_center = None
    best_dist = 1e9
    for cx, cy in centers:
        dist = (cx - px) ** 2 + (cy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best_center = [cx, cy]
    return best_center
