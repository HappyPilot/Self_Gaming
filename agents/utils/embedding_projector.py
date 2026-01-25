#!/usr/bin/env python3
"""Lightweight random projector for visual embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _make_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mat = rng.normal(0.0, 1.0, size=(in_dim, out_dim)).astype(np.float32)
    # Orthonormalize columns for stable cosine geometry.
    q, _ = np.linalg.qr(mat)
    return q[:, :out_dim].astype(np.float32)


def _layer_norm(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(vec.mean())
    var = float(vec.var())
    return (vec - mean) / float(np.sqrt(var + eps))


@dataclass
class EmbeddingProjector:
    in_dim: int
    out_dim: int
    seed: int = 42
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        self._proj = _make_projection(self.in_dim, self.out_dim, self.seed)

    def project(self, embedding: Optional[list]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size != self.in_dim:
            return None
        out = arr @ self._proj
        if self.use_layer_norm:
            out = _layer_norm(out)
        return out.astype(np.float32)
