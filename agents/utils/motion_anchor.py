from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _apply_ui_mask(mag: np.ndarray, mask_bottom: float, mask_corner_x: float) -> np.ndarray:
    h, w = mag.shape[:2]
    if mask_bottom > 0:
        y0 = int(h * (1.0 - mask_bottom))
        mag[y0:, :] = 0
    if mask_corner_x > 0:
        xw = int(w * mask_corner_x)
        if xw > 0:
            mag[:, :xw] = 0
            mag[:, w - xw :] = 0
    return mag


def compute_motion_center(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    mag_threshold: float,
    min_mean: float,
    mask_bottom: float = 0.0,
    mask_corner_x: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    if prev_gray is None or curr_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        return None
    diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
    diff = _apply_ui_mask(diff, mask_bottom, mask_corner_x)
    mean_val = float(diff.mean()) if diff.size else 0.0
    if mean_val < min_mean:
        return None
    mask = diff >= mag_threshold
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    weights = diff[ys, xs]
    total = float(weights.sum()) if weights.size else 0.0
    if total <= 0.0:
        return None
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    h, w = diff.shape[:2]
    x_norm = max(0.0, min(1.0, cx / max(1.0, w - 1)))
    y_norm = max(0.0, min(1.0, cy / max(1.0, h - 1)))
    score = mean_val
    return x_norm, y_norm, score
