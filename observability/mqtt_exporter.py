"""Prometheus exporter that bridges MQTT metrics to /metrics."""
from __future__ import annotations

import json
import logging
import os
import time

try:
    import paho.mqtt.client as mqtt
except Exception:  # noqa: BLE001
    mqtt = None

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except Exception:  # noqa: BLE001
    Counter = Gauge = Histogram = start_http_server = None

LOG_LEVEL = os.getenv("EXPORTER_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("mqtt_metrics_exporter")

MQTT_HOST = os.getenv("MQTT_HOST", "mq")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "30"))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "metrics_exporter")

LATENCY_TOPIC = os.getenv("LATENCY_TOPIC", "metrics/latency")
CONTROL_TOPIC = os.getenv("CONTROL_METRIC_TOPIC", "metrics/control")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "9108"))


def _require_deps() -> bool:
    if mqtt is None:
        logger.error("paho-mqtt is not installed; metrics exporter cannot start")
        return False
    if Counter is None or Gauge is None or Histogram is None or start_http_server is None:
        logger.error("prometheus_client is not installed; metrics exporter cannot start")
        return False
    return True


LATENCY_HIST = None
LATENCY_VIOLATIONS = None
CONTROL_VALUE = None
CONTROL_FAILS = None
EXPORTER_ERRORS = None
LAST_SEEN = None


def _init_metrics() -> None:
    global LATENCY_HIST, LATENCY_VIOLATIONS, CONTROL_VALUE, CONTROL_FAILS, EXPORTER_ERRORS, LAST_SEEN
    LATENCY_HIST = Histogram(
        "self_gaming_latency_ms",
        "Latency in milliseconds",
        ["stage", "agent"],
    )
    LATENCY_VIOLATIONS = Counter(
        "self_gaming_latency_violations_total",
        "Latency SLA violations",
        ["stage", "agent"],
    )
    CONTROL_VALUE = Gauge(
        "self_gaming_control_metric_value",
        "Control metric value",
        ["metric", "agent"],
    )
    CONTROL_FAILS = Counter(
        "self_gaming_control_metric_failures_total",
        "Control metric failures",
        ["metric", "agent"],
    )
    EXPORTER_ERRORS = Counter(
        "self_gaming_exporter_errors_total",
        "Exporter parse errors",
        ["topic", "reason"],
    )
    LAST_SEEN = Gauge(
        "self_gaming_mqtt_last_seen_seconds",
        "Last seen MQTT event timestamp",
        ["topic"],
    )


def _as_agent(payload: dict) -> str:
    agent = payload.get("agent")
    if agent:
        return str(agent)
    tags = payload.get("tags")
    if isinstance(tags, dict) and tags.get("agent"):
        return str(tags.get("agent"))
    return "unknown"


def _handle_latency(payload: dict) -> None:
    stage = payload.get("stage")
    duration_ms = payload.get("duration_ms")
    if stage is None or duration_ms is None:
        EXPORTER_ERRORS.labels(LATENCY_TOPIC, "missing_fields").inc()
        return
    agent = _as_agent(payload)
    try:
        LATENCY_HIST.labels(str(stage), agent).observe(float(duration_ms))
    except Exception:
        EXPORTER_ERRORS.labels(LATENCY_TOPIC, "observe_failed").inc()
        return
    ok = payload.get("ok")
    if ok is False:
        LATENCY_VIOLATIONS.labels(str(stage), agent).inc()
    ts = payload.get("timestamp") or time.time()
    LAST_SEEN.labels(LATENCY_TOPIC).set(float(ts))


def _handle_control(payload: dict) -> None:
    metric = payload.get("metric")
    value = payload.get("value")
    if metric is None or value is None:
        EXPORTER_ERRORS.labels(CONTROL_TOPIC, "missing_fields").inc()
        return
    agent = _as_agent(payload)
    try:
        CONTROL_VALUE.labels(str(metric), agent).set(float(value))
    except Exception:
        EXPORTER_ERRORS.labels(CONTROL_TOPIC, "set_failed").inc()
        return
    ok = payload.get("ok")
    if ok is False:
        CONTROL_FAILS.labels(str(metric), agent).inc()
    ts = payload.get("timestamp") or time.time()
    LAST_SEEN.labels(CONTROL_TOPIC).set(float(ts))


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe([(LATENCY_TOPIC, 0), (CONTROL_TOPIC, 0)])
        logger.info("Connected to MQTT; subscribed to %s, %s", LATENCY_TOPIC, CONTROL_TOPIC)
    else:
        logger.error("MQTT connect failed rc=%s", rc)


def _on_message(_client, _userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        EXPORTER_ERRORS.labels(msg.topic, "decode_failed").inc()
        return
    if not isinstance(payload, dict):
        EXPORTER_ERRORS.labels(msg.topic, "not_dict").inc()
        return
    if msg.topic == LATENCY_TOPIC:
        _handle_latency(payload)
    elif msg.topic == CONTROL_TOPIC:
        _handle_control(payload)


def main() -> None:
    if not _require_deps():
        raise SystemExit(1)
    _init_metrics()
    start_http_server(EXPORTER_PORT)
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)
    client.on_connect = _on_connect
    client.on_message = _on_message
    try:
        client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("MQTT connection failed: %s", exc)
        raise SystemExit(2)
    client.loop_forever()


if __name__ == "__main__":
    main()
