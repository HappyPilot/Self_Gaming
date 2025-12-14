"""Helpers for reporting curriculum status based on mem_agent data."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_SCOPE = os.getenv("CURRIC_DEATH_SCOPE", "critical_dialog:death")
DEFAULT_RULE_LIMIT = int(os.getenv("CURRIC_DEATH_RULES_LIMIT", "5"))
DEFAULT_RECENT_LIMIT = int(os.getenv("CURRIC_DEATH_RECENT_LIMIT", "20"))
DEFAULT_CAL_LIMIT = int(os.getenv("CURRIC_DEATH_CAL_LIMIT", "5"))
DEFAULT_MIN_RESURRECTS = int(os.getenv("CURRIC_DEATH_MIN_RESURRECTS", "5"))
DEFAULT_MIN_SUCCESS_RATE = float(os.getenv("CURRIC_DEATH_MIN_SUCCESS_RATE", "0.6"))
DEFAULT_MIN_CALIBRATIONS = int(os.getenv("CURRIC_DEATH_MIN_CALIBRATIONS", "1"))
DEFAULT_TIMEOUT = float(os.getenv("CURRIC_DEATH_MEM_TIMEOUT", "1.0"))


class CurriculumStatusError(RuntimeError):
    """Raised when curriculum readiness cannot be computed."""


@dataclass
class DeathCurriculumStatus:
    scope: str
    profile: Optional[str]
    death_count: int
    resurrect_count: int
    other_count: int
    success_rate: Optional[float]
    rules_count: int
    calibration_count: int
    ready: bool
    reason: str


def _safe_query(mem_rpc, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    response = mem_rpc.query(payload, timeout=timeout)
    if not response or not isinstance(response, dict):
        raise CurriculumStatusError("empty mem_agent response")
    return response


def compute_death_curriculum_status(
    mem_rpc,
    scope: str = DEFAULT_SCOPE,
    profile: Optional[str] = None,
    *,
    rule_limit: int = DEFAULT_RULE_LIMIT,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    calibration_limit: int = DEFAULT_CAL_LIMIT,
    min_resurrects: int = DEFAULT_MIN_RESURRECTS,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    min_calibrations: int = DEFAULT_MIN_CALIBRATIONS,
    timeout: float = DEFAULT_TIMEOUT,
) -> DeathCurriculumStatus:
    """Compute readiness for the death dialog curriculum.

    Parameters mirror the CURRIC_DEATH_* environment variables and allow
    callers to override limits or thresholds explicitly when desired.
    """

    if mem_rpc is None:
        raise CurriculumStatusError("mem_rpc unavailable")

    rules_payload = {"mode": "rules", "scope": scope, "limit": rule_limit}
    rules_resp = _safe_query(mem_rpc, rules_payload, timeout)
    rules = rules_resp.get("value") or []

    recent_payload = {"mode": "recent_critical", "scope": scope, "limit": recent_limit}
    recent_resp = _safe_query(mem_rpc, recent_payload, timeout)
    recent_entries = recent_resp.get("value") or []
    death_count = sum(1 for entry in recent_entries if entry.get("delta") == "hero_dead")
    resurrect_count = sum(
        1 for entry in recent_entries if entry.get("delta") == "hero_resurrected"
    )
    other_count = max(0, len(recent_entries) - death_count - resurrect_count)

    calib_payload = {
        "mode": "calibration_events",
        "scope": scope,
        "profile": profile,
        "limit": calibration_limit,
    }
    calib_resp = _safe_query(mem_rpc, calib_payload, timeout)
    calibrations = calib_resp.get("value") or []

    success_rate: Optional[float]
    if death_count > 0:
        rate = resurrect_count / float(death_count)
        success_rate = min(1.0, round(rate, 3))
    else:
        success_rate = None

    calibration_count = len(calibrations)
    rules_count = len(rules)

    ready = True
    reason = "ready"

    if calibration_count < min_calibrations:
        ready = False
        reason = "no_calibration"
    elif death_count == 0:
        ready = False
        reason = "insufficient_data"
    elif resurrect_count < min_resurrects:
        ready = False
        reason = "insufficient_resurrections"
    elif success_rate is None or success_rate < min_success_rate:
        ready = False
        reason = "low_success_rate"

    return DeathCurriculumStatus(
        scope=scope,
        profile=profile,
        death_count=death_count,
        resurrect_count=resurrect_count,
        other_count=other_count,
        success_rate=success_rate,
        rules_count=rules_count,
        calibration_count=calibration_count,
        ready=ready,
        reason=reason,
    )


__all__ = [
    "CurriculumStatusError",
    "DeathCurriculumStatus",
    "compute_death_curriculum_status",
]

