"""Lightweight frame encoder for world-state adapters."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


DEFAULT_FRAME_SIZE = int(os.getenv("WORLD_STATE_FRAME_SIZE", "64"))
DEFAULT_LATENT_DIM = int(os.getenv("WORLD_STATE_LATENT_DIM", "128"))
DEFAULT_SEED = int(os.getenv("WORLD_STATE_SEED", "1337"))
DEFAULT_NORMALIZE = os.getenv("WORLD_STATE_NORMALIZE", "1") != "0"


class FrameEncoder:
    """Downscale + optional random projection to produce a cheap latent vector."""

    def __init__(
        self,
        frame_size: int = DEFAULT_FRAME_SIZE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        seed: int = DEFAULT_SEED,
        normalize: bool = DEFAULT_NORMALIZE,
    ) -> None:
        self.frame_size = max(8, int(frame_size))
        self.latent_dim = max(0, int(latent_dim))
        self.normalize = normalize
        self._rng = np.random.RandomState(seed)
        self._proj: Optional[np.ndarray] = None

    def encode_frame(self, frame: np.ndarray | None) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        small = self._downscale(frame, self.frame_size)
        if small is None or small.size == 0:
            return None
        flat = small.astype(np.float32).reshape(-1)
        if self.normalize and flat.size:
            flat /= 255.0
        return self._project(flat)

    def encode_bytes(self, data: bytes) -> Optional[np.ndarray]:
        if not data or cv2 is None:
            return None
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return self.encode_frame(frame)

    def _project(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if self.latent_dim <= 0:
            return None
        if self.latent_dim >= vector.size:
            return vector
        if self._proj is None or self._proj.shape[0] != vector.size:
            scale = 1.0 / max(1.0, vector.size) ** 0.5
            self._proj = self._rng.normal(scale=scale, size=(vector.size, self.latent_dim)).astype(np.float32)
        return vector @ self._proj

    def _downscale(self, frame: np.ndarray, size: int) -> Optional[np.ndarray]:
        if cv2 is not None:
            resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        else:
            ys = np.linspace(0, frame.shape[0] - 1, num=size, dtype=int)
            xs = np.linspace(0, frame.shape[1] - 1, num=size, dtype=int)
            resized = frame[np.ix_(ys, xs)]
        if resized.ndim == 3:
            resized = resized.mean(axis=2)
        return resized
