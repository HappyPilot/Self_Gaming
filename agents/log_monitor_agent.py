#!/usr/bin/env python3
"""Aggregates agent log files and emits training-health alerts over MQTT."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import signal
import threading

import paho.mqtt.client as mqtt

logger = logging.getLogger("log_monitor")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SUMMARY_TOPIC = os.getenv("LOG_SUMMARY_TOPIC", "logs/summary")
ALERT_TOPIC = os.getenv("LOG_ALERT_TOPIC", "logs/alerts")
SCAN_INTERVAL = float(os.getenv("LOG_MONITOR_INTERVAL", "5.0"))
LOG_DIRS_RAW = os.getenv("LOG_MONITOR_DIRS") or os.getenv("LOG_MONITOR_DIR") or os.getenv("LOG_DIR")
if LOG_DIRS_RAW:
    LOG_DIRS = [Path(entry.strip()) for entry in LOG_DIRS_RAW.split(":") if entry.strip()]
else:
    LOG_DIRS = [Path("/app/logs")]

ALERT_SILENCE_SEC = float(os.getenv("LOG_ALERT_COOLDOWN", "30"))
TRAINING_SOURCES = {src.strip() for src in os.getenv("LOG_MONITOR_TRAINERS", "train_manager,policy_agent,teach_agent").split(",") if src.strip()}

ERROR_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bERROR\b", re.IGNORECASE), "error"),
    (re.compile(r"\bCRITICAL\b", re.IGNORECASE), "critical"),
    (re.compile(r"traceback", re.IGNORECASE), "traceback"),
)
TRAINING_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"job_failed", re.IGNORECASE), "train_job_failed"),
    (re.compile(r"nan", re.IGNORECASE), "nan_values"),
    (re.compile(r"loss .*nan", re.IGNORECASE), "loss_nan"),
    (re.compile(r"reward\s*[:=]\s*-?0\.[5-9]+", re.IGNORECASE), "low_reward"),
)

stop_event = threading.Event()


def _as_int(code) -> int:
    """Safely convert a paho-mqtt v2 reason code to int."""
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


@dataclass
class FileState:
    path: Path
    position: int = 0
    inode: int = 0
    stats: Dict[str, int] = field(default_factory=lambda: {"errors": 0, "warnings": 0, "lines": 0})


class LogMonitorAgent:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="log_monitor_agent")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.states: Dict[Path, FileState] = {}
        self.alert_memory: Dict[str, float] = {}

    # MQTT -------------------------------------------------------------
    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            logger.info("Log monitor connected to MQTT")
            client.publish(SUMMARY_TOPIC, json.dumps({"ok": True, "event": "log_monitor_ready"}))
        else:
            logger.error("Log monitor failed to connect rc=%s", _as_int(rc))

    def _on_disconnect(self, _client, _userdata, rc):
        logger.warning("Log monitor disconnected rc=%s", _as_int(rc))

    # Scanning ---------------------------------------------------------
    def _iter_log_files(self) -> Iterable[Path]:
        for directory in LOG_DIRS:
            if not directory.exists():
                continue
            for file in directory.glob("*.log"):
                if file.is_file():
                    yield file

    def _rotation_state(self, state: FileState, stat_result) -> None:
        inode = getattr(stat_result, "st_ino", None)
        if inode is None:
            return
        if state.inode != inode:
            state.inode = inode
            state.position = 0

    def _process_line(self, line: str, source: str) -> Tuple[str, List[str]]:
        severity = "info"
        issues: List[str] = []
        if "WARNING" in line.upper():
            severity = "warning"
        for pattern, tag in ERROR_PATTERNS:
            if pattern.search(line):
                severity = "error"
                issues.append(tag)
        if source in TRAINING_SOURCES:
            for pattern, tag in TRAINING_PATTERNS:
                if pattern.search(line):
                    issues.append(tag)
        return severity, issues

    def _scan_file(self, path: Path) -> Tuple[Dict[str, int], List[dict]]:
        state = self.states.setdefault(path, FileState(path=path))
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            return state.stats, []
        self._rotation_state(state, stat_result)
        alerts: List[dict] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(state.position)
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    state.stats["lines"] += 1
                    severity, issues = self._process_line(line, path.stem)
                    if severity == "error":
                        state.stats["errors"] += 1
                    elif severity == "warning":
                        state.stats["warnings"] += 1
                    for issue in issues:
                        alerts.append({
                            "source": path.stem,
                            "issue": issue,
                            "line": line[-400:],
                        })
                state.position = handle.tell()
        except (OSError, IOError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
        return state.stats.copy(), alerts

    def scan_once(self) -> Tuple[Dict[str, dict], List[dict]]:
        summary: Dict[str, dict] = {}
        alerts: List[dict] = []
        for logfile in self._iter_log_files():
            stats, file_alerts = self._scan_file(logfile)
            if stats["lines"]:
                summary[logfile.stem] = stats
            for alert in file_alerts:
                alert["timestamp"] = time.time()
                alerts.append(alert)
        return summary, alerts

    # Publishing -------------------------------------------------------
    def _emit_summary(self, summary: Dict[str, dict]):
        payload = {
            "ok": True,
            "timestamp": time.time(),
            "sources": summary,
        }
        self.client.publish(SUMMARY_TOPIC, json.dumps(payload))

    def _emit_alert(self, alert: dict):
        key = f"{alert['source']}::{alert['issue']}"
        now = time.time()
        last = self.alert_memory.get(key, 0.0)
        if now - last < ALERT_SILENCE_SEC:
            return
        self.alert_memory[key] = now
        payload = {
            "ok": True,
            "timestamp": now,
            **alert,
        }
        self.client.publish(ALERT_TOPIC, json.dumps(payload))

    # Main loop -------------------------------------------------------
    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()
        try:
            while not stop_event.is_set():
                summary, alerts = self.scan_once()
                if summary:
                    self._emit_summary(summary)
                for alert in alerts:
                    self._emit_alert(alert)
                time.sleep(SCAN_INTERVAL)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def _handle_signal(_signum, _frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = LogMonitorAgent()
    agent.start()


if __name__ == "__main__":
    main()
