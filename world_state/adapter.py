"""World state adapter that fuses frame latents + sensors."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    from utils.frame_transport import get_frame_bytes
except Exception:  # noqa: BLE001
    from agents.utils.frame_transport import get_frame_bytes
from world_state.encoder import FrameEncoder


class WorldStateAdapter:
    """Builds a lightweight state dict from frames and observation payloads."""

    def __init__(self, encoder: Optional[FrameEncoder] = None, include_raw: bool = False) -> None:
        self.encoder = encoder or FrameEncoder()
        self.include_raw = include_raw

    def build_state(
        self,
        *,
        observation: Optional[Dict[str, Any]] = None,
        frame: Optional[np.ndarray] = None,
        frame_payload: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs = observation or {}
        latent = None
        if frame_payload is not None and frame is None:
            data = get_frame_bytes(frame_payload)
            if data:
                latent = self.encoder.encode_bytes(data)
        if latent is None and frame is not None:
            latent = self.encoder.encode_frame(frame)
        objects = obs.get("yolo_objects") or obs.get("objects") or []
        text_zones = obs.get("text_zones") or {}
        text = [zone.get("text") for zone in text_zones.values() if isinstance(zone, dict) and zone.get("text")]
        player = obs.get("player_candidate") or obs.get("player")
        state = {
            "timestamp": float(timestamp or obs.get("timestamp") or time.time()),
            "frame_id": obs.get("frame_id"),
            "latent": latent.tolist() if isinstance(latent, np.ndarray) else None,
            "objects": objects,
            "object_count": len(objects) if isinstance(objects, list) else 0,
            "text": text,
            "text_count": len(text),
            "player": player,
            "reward": reward,
        }
        if self.include_raw:
            state["observation"] = obs
        return state
