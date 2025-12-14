#!/usr/bin/env python3
"""CLI summary of memory rules and recent critical episodes."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.mem_rpc import MemRPC  # noqa: E402

DEFAULT_RULE_LIMIT = int(os.getenv("MEM_STATUS_RULE_LIMIT", "5"))
DEFAULT_CRITICAL_LIMIT = int(os.getenv("MEM_STATUS_CRITICAL_LIMIT", "5"))
DEFAULT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("MQTT_PORT", "1883"))


def human_time(ts: float | None) -> str:
    if not ts:
        return "n/a"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def format_rules(rules: List[dict], limit: int) -> str:
    if not rules:
        return "Rules: none"
    sorted_rules = sorted(
        rules,
        key=lambda r: (-int(r.get("usage_count", 0)), -(r.get("last_used_at") or 0.0)),
    )
    lines = ["Rules (top %s):" % min(limit, len(sorted_rules))]
    for idx, rule in enumerate(sorted_rules[:limit], 1):
        lines.append(
            f"  {idx}. [{rule.get('scope')}] {rule.get('text','')[:80]}"
            f" (usage={rule.get('usage_count',0)}, confidence={rule.get('confidence',0):.2f}, last_used={human_time(rule.get('last_used_at'))})"
        )
    return "\n".join(lines)


def _recent_counts(entries: List[dict]) -> tuple[int, int, int]:
    deaths = sum(1 for entry in entries if entry.get("delta") == "hero_dead")
    resurrected = sum(1 for entry in entries if entry.get("delta") == "hero_resurrected")
    other = max(0, len(entries) - deaths - resurrected)
    return deaths, resurrected, other


def format_recent_critical(entries: List[dict], limit: int) -> str:
    if not entries:
        return "Recent critical: none"
    lines = ["Recent critical (last %s):" % min(limit, len(entries))]
    for entry in entries[:limit]:
        lines.append(
            f"  - {human_time(entry.get('timestamp'))} {entry.get('scope')}"
            f" delta={entry.get('delta')} episode={entry.get('episode_id')} tags={entry.get('tags')}"
        )
    deaths, resurrected, other = _recent_counts(entries)
    lines.append(
        f"  stats: hero_dead={deaths} hero_resurrected={resurrected} other={other}"
    )
    return "\n".join(lines)


def format_calibrations(entries: List[dict], limit: int, scope: str, profile: str) -> str:
    if not entries:
        return f"Calibration events: none for scope={scope or '*'} profile={profile or '*'}"
    lines = [
        "Calibration events (last %s for scope=%s profile=%s):"
        % (min(limit, len(entries)), scope or "*", profile or "*")
    ]
    for entry in entries[:limit]:
        try:
            x_val = float(entry.get("x_norm")) if entry.get("x_norm") is not None else None
        except (TypeError, ValueError):
            x_val = None
        try:
            y_val = float(entry.get("y_norm")) if entry.get("y_norm") is not None else None
        except (TypeError, ValueError):
            y_val = None
        x_text = f"{x_val:.3f}" if x_val is not None else "n/a"
        y_text = f"{y_val:.3f}" if y_val is not None else "n/a"
        lines.append(
            f"  - {human_time(entry.get('timestamp'))} coords=({x_text}, {y_text})"
            f" profile={entry.get('profile')} scope={entry.get('scope')}"
        )
    return "\n".join(lines)


def build_summary_text(
    rules: List[dict],
    critical: List[dict],
    calibrations: List[dict],
    rule_limit: int,
    critical_limit: int,
    cal_limit: int,
    scope: str,
    profile: str,
) -> str:
    sections = [
        format_rules(rules, rule_limit),
        format_recent_critical(critical, critical_limit),
        format_calibrations(calibrations, cal_limit, scope, profile),
    ]
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Show memory status")
    parser.add_argument("--scope", default="", help="Scope to query (e.g. critical_dialog:death)")
    parser.add_argument("--profile", default="", help="Dialog profile (for calibration stats)")
    parser.add_argument("--rules-limit", type=int, default=DEFAULT_RULE_LIMIT)
    parser.add_argument("--critical-limit", type=int, default=DEFAULT_CRITICAL_LIMIT)
    parser.add_argument("--cal-limit", type=int, default=3, help="Calibration events to show")
    parser.add_argument("--host", default=DEFAULT_HOST, help="MQTT host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    try:
        mem = MemRPC(host=args.host, port=args.port)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to connect to mem_agent via MQTT: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        scope = args.scope
        rule_payload = {"mode": "rules", "scope": scope, "limit": args.rules_limit}
        rule_response = mem.query(rule_payload, timeout=1.0) or {}
        rules = rule_response.get("value") or []

        crit_payload = {"mode": "recent_critical", "scope": scope, "limit": args.critical_limit}
        crit_response = mem.query(crit_payload, timeout=1.0) or {}
        critical = crit_response.get("value") or []
        cal_payload = {
            "mode": "calibration_events",
            "scope": scope,
            "profile": args.profile,
            "limit": args.cal_limit,
        }
        cal_response = mem.query(cal_payload, timeout=1.0) or {}
        calibrations = cal_response.get("value") or []

        print(
            build_summary_text(
                rules,
                critical,
                calibrations,
                args.rules_limit,
                args.critical_limit,
                args.cal_limit,
                scope,
                args.profile,
            )
        )
    finally:
        mem.close()


if __name__ == "__main__":
    main()
