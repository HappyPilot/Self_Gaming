"""Latency helpers for publishing metrics/latency events."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

LATENCY_TOPIC = os.getenv("LATENCY_TOPIC", "metrics/latency")


def _parse_float_env(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_sla_ms(name: str, default: Optional[float] = None) -> Optional[float]:
    return _parse_float_env(name, default)


def build_latency_event(
    stage: str,
    duration_ms: float,
    *,
    sla_ms: Optional[float] = None,
    tags: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": "latency",
        "stage": stage,
        "duration_ms": round(float(duration_ms), 3),
        "timestamp": float(timestamp) if timestamp is not None else time.time(),
    }
    if sla_ms is not None:
        payload["sla_ms"] = float(sla_ms)
        payload["ok"] = payload["duration_ms"] <= float(sla_ms)
    if agent:
        payload["agent"] = agent
    if trace_id:
        payload["trace_id"] = trace_id
    if span_id:
        payload["span_id"] = span_id
    if tags:
        payload["tags"] = tags
    return payload


def emit_latency(
    client,
    stage: str,
    duration_ms: float,
    *,
    sla_ms: Optional[float] = None,
    tags: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None,
    topic: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    payload = build_latency_event(stage, duration_ms, sla_ms=sla_ms, tags=tags, agent=agent)
    try:
        client.publish(topic or LATENCY_TOPIC, json.dumps(payload))
    except Exception:
        return None
    return payload


@contextmanager
def latency_timer(
    client,
    stage: str,
    *,
    sla_ms: Optional[float] = None,
    tags: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None,
    topic: Optional[str] = None,
):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        emit_latency(client, stage, duration_ms, sla_ms=sla_ms, tags=tags, agent=agent, topic=topic)
