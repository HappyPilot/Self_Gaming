"""Policy adapter factory for selecting backend implementations."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("policy_adapter_factory")


def create_policy_adapter(
    action_space_dim: int,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Optional[object]:
    backend_value = (backend or os.getenv("POLICY_ADAPTER_BACKEND", "reflex")).strip().lower()
    if not backend_value:
        backend_value = "reflex"
    if backend_value == "reflex":
        return None
    if backend_value == "titans":
        try:
            from sg_platform.policy_titans_adapter import TitansPolicyAdapter

            return TitansPolicyAdapter(action_space_dim=action_space_dim, device=device, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Titans backend unavailable, falling back to reflex: %s", exc)
            return None
    logger.warning("Unknown policy adapter backend %s; falling back to reflex", backend_value)
    return None
