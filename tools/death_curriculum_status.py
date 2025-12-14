#!/usr/bin/env python3
"""CLI to print the death-dialog curriculum readiness status."""
from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.mem_rpc import MemRPC  # noqa: E402
from agents.curriculum_status import (  # noqa: E402
    CurriculumStatusError,
    compute_death_curriculum_status,
)

DEFAULT_SCOPE = os.getenv("CURRIC_DEATH_SCOPE", "critical_dialog:death")
DEFAULT_PROFILE = os.getenv("CURRIC_DEATH_PROFILE") or os.getenv("GOAP_DIALOG_PROFILE")


def format_bool(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> int:
    parser = argparse.ArgumentParser(description="Show death curriculum readiness status")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="mem scope (default: %(default)s)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Dialog profile (optional)")
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    args = parser.parse_args()

    try:
        mem = MemRPC(host=args.host, port=args.port)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to connect to mem_agent MQTT bridge: {exc}", file=sys.stderr)
        return 0

    try:
        status = compute_death_curriculum_status(mem, scope=args.scope, profile=args.profile)
    except CurriculumStatusError as exc:
        print(f"Unable to compute curriculum status: {exc}", file=sys.stderr)
        return 0
    finally:
        mem.close()

    print(f"Scope        : {status.scope}")
    print(f"Profile      : {status.profile or '(default)'}")
    print(f"Deaths       : {status.death_count}")
    print(f"Resurrects   : {status.resurrect_count}")
    print(f"Other events : {status.other_count}")
    sr_text = "n/a" if status.success_rate is None else f"{status.success_rate:.2f}"
    print(f"Success rate : {sr_text}")
    print(f"Rules cached : {status.rules_count}")
    print(f"Calibrations : {status.calibration_count}")
    print(f"READY        : {format_bool(status.ready)} (reason={status.reason})")
    if not status.ready:
        print("NOTE: requirements for advancing the death curriculum are not met yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
