"""Universal Env API contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Observation:
    timestamp: float
    frame_id: Optional[int]
    payload: Dict[str, Any]
    topic: str = ""


@dataclass(frozen=True)
class Action:
    payload: Dict[str, Any]
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class StepResult:
    observation: Optional[Observation]
    reward: Optional[float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)


class EnvAdapter(ABC):
    """Gym-style interface for environment interaction."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset or resync the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: Dict[str, Any] | Action) -> StepResult:
        """Apply an action and return the resulting StepResult."""

    @abstractmethod
    def get_observation(self, timeout_sec: Optional[float] = None) -> Optional[Observation]:
        """Fetch the latest observation (optionally waiting up to timeout)."""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return a health status dict for the environment."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources."""
