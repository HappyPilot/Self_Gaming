#!/usr/bin/env python3
"""Tiny scene fingerprint helper for gating repeated frames."""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def compute_fingerprint(img: np.ndarray, target_w: int = 64) -> str:
    """Compute md5 hash of a downscaled grayscale frame."""
    if img is None:
        return ""
    import cv2  # local import to avoid hard dependency if unused

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return ""
    scale = target_w / float(w)
    target_h = max(1, int(h * scale))
    small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return hashlib.md5(gray.tobytes()).hexdigest()
