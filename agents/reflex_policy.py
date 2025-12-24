"""Reflex policy adapter scaffold."""
from __future__ import annotations

from typing import Any, Dict, Optional


class PolicyAdapter:
    """Minimal policy adapter interface."""

    def predict(self, observation: Dict[str, Any], strategy_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ReflexPolicyAdapter(PolicyAdapter):
    """Simple reflex policy placeholder that emits no-op actions."""

    def predict(self, observation: Dict[str, Any], strategy_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None
