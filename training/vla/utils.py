"""Helper utilities for VLA training scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def has_click(payload: Dict[str, Any]) -> bool:
    if "click" in payload:
        value = payload.get("click")
        if isinstance(value, bool):
            return value
        return value is not None
    if "button" in payload:
        value = payload.get("button")
        if isinstance(value, bool):
            return value
        return value is not None
    if "buttons" in payload:
        value = payload.get("buttons")
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        if isinstance(value, bool):
            return value
        return value is not None
    return False


def encode_action(payload: Dict[str, Any], action_dim: int) -> List[float]:
    dx = float(payload.get("dx", 0.0) or 0.0)
    dy = float(payload.get("dy", 0.0) or 0.0)
    click = has_click(payload)
    key = payload.get("key") or payload.get("keys")
    vector = [dx, dy, 1.0 if click else 0.0, 1.0 if key else 0.0]
    if action_dim <= 0:
        return []
    if action_dim == len(vector):
        return vector
    if action_dim < len(vector):
        return vector[:action_dim]
    return vector + [0.0] * (action_dim - len(vector))
