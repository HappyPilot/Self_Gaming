"""Reflex policy adapter scaffold."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from sg_platform.policy_adapter_factory import create_policy_adapter

logger = logging.getLogger("reflex_policy")


class PolicyAdapter:
    """Minimal policy adapter interface."""

    def predict(self, observation: Dict[str, Any], strategy_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ReflexPolicyAdapter(PolicyAdapter):
    """Simple reflex policy placeholder that emits no-op actions."""

    def __init__(self) -> None:
        backend = os.getenv("POLICY_ADAPTER_BACKEND", "reflex").strip().lower() or "reflex"
        action_space_dim = int(os.getenv("POLICY_ACTION_DIM", "2"))
        self.adapter = create_policy_adapter(action_space_dim=action_space_dim)
        logger.info("PolicyAdapter backend=%s", backend)

    def predict(self, observation: Dict[str, Any], strategy_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.adapter is None:
            return None
        if hasattr(self.adapter, "predict"):
            return self.adapter.predict(observation, strategy_state)
        if hasattr(self.adapter, "predict_chunk"):
            chunk = self.adapter.predict_chunk(observation, strategy_state)
            if isinstance(chunk, dict):
                actions = chunk.get("actions")
                if isinstance(actions, list) and actions:
                    action = actions[0]
                    if isinstance(action, dict):
                        return action
        return None
