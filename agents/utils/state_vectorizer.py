#!/usr/bin/env python3
"""Deterministic feature hashing vectorizer for scene/state JSON."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _hash(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % size


def _flatten(obj, prefix: str = "") -> Iterable[Tuple[str, str]]:
    """Yield (path, str_value) pairs from nested JSON-like objects."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten(v, new_prefix)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            yield from _flatten(v, new_prefix)
    else:
        yield prefix, str(obj)


class StateVectorizer:
    def __init__(self, size: int = 512, schema_version: int = 1):
        self.size = size
        self.schema_version = schema_version

    def vectorize(self, state: Dict) -> np.ndarray:
        vec = np.zeros(self.size, dtype=np.float32)
        for path, value in _flatten(state):
            if not path:
                continue
            # numeric handling
            try:
                num = float(value)
                idx = _hash(path + ":num", self.size)
                vec[idx] += math.tanh(num)
                continue
            except Exception:
                pass
            # tokenize string
            tokens: List[str] = str(value).lower().split()
            for tok in tokens:
                idx = _hash(path + ":" + tok, self.size)
                vec[idx] += 1.0
        return vec
