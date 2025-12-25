"""In-memory EnvAdapter for tests and simulations."""
from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional

from env_api.adapter import Action, EnvAdapter, Observation, StepResult


class InMemoryEnvAdapter(EnvAdapter):
    """Queue-backed EnvAdapter for unit tests and local simulations."""

    def __init__(self) -> None:
        self._obs_queue: queue.Queue[Observation] = queue.Queue()
        self._latest_obs: Optional[Observation] = None

    def push_observation(self, payload: Dict[str, Any], topic: str = "vision/observation") -> Observation:
        ts = float(payload.get("timestamp", time.time()))
        frame_id = payload.get("frame_id")
        try:
            frame_id = int(frame_id) if frame_id is not None else None
        except (TypeError, ValueError):
            frame_id = None
        obs = Observation(timestamp=ts, frame_id=frame_id, payload=payload, topic=topic)
        self._latest_obs = obs
        self._obs_queue.put(obs)
        return obs

    def reset(self) -> Observation:
        obs = self.get_observation(timeout_sec=0.1)
        if obs is None:
            raise RuntimeError("No observation available for reset()")
        return obs

    def step(self, action: Dict[str, Any] | Action) -> StepResult:
        obs = self.get_observation(timeout_sec=0.1)
        return StepResult(obs, None, False, info={"action": _coerce_action(action)})

    def get_observation(self, timeout_sec: Optional[float] = None) -> Optional[Observation]:
        try:
            timeout = 0.1 if timeout_sec is None else timeout_sec
            return self._obs_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "in_memory": True, "last_obs": self._latest_obs is not None}

    def close(self) -> None:
        return None


def _coerce_action(action: Dict[str, Any] | Action) -> Dict[str, Any]:
    if isinstance(action, Action):
        payload = dict(action.payload)
        if action.timestamp is not None:
            payload.setdefault("timestamp", action.timestamp)
        return payload
    return dict(action)
