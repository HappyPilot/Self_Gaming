"""Global logging configuration for every agent.

This module is imported automatically by Python when present on the module
search path.  We use it to attach a rotating file handler (plus the default
stdout handler) for *all* agents without touching each script individually.
The handler destination is selected from the following locations (first
writable wins):

1. The LOG_DIR environment variable, if set.
2. /mnt/ssd/logs (preferred target on the Jetson)
3. /app/logs (bind-mounted repo inside the containers)
4. ~/Documents/agent_reports (fallback on macOS dev machines)
5. ./logs inside the repo
6. ~/agent_logs
7. /tmp/agent_logs
"""
from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_LOCATIONS = [
    os.getenv("LOG_DIR"),
    Path("/mnt/ssd/logs"),
    Path("/app/logs"),
    Path.home() / "Documents" / "agent_reports",
    _BASE_DIR / "logs",
    Path.home() / "agent_logs",
    Path("/tmp/agent_logs"),
]
_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
_STREAM_ENABLED = os.getenv("LOG_STREAM", "1") not in {"0", "false", "False"}


def _first_writable(paths: Iterable[Optional[str]]) -> Optional[Path]:
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        else:
            return path
    return None


def _resolve_logfile() -> Optional[Path]:
    target_dir = _first_writable(_DEFAULT_LOCATIONS)
    if target_dir is None:
        return None
    agent_name = (
        os.getenv("AGENT_NAME")
        or os.getenv("APP_LOG_NAME")
        or Path(sys.argv[0]).stem
        or "agent"
    )
    safe_name = agent_name.replace("/", "_")
    return target_dir / f"{safe_name}.log"


def _configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_global_log_configured", False):
        return

    logfile = _resolve_logfile()
    handlers = []
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s %(message)s"
    )
    formatter.converter = time.gmtime

    if logfile:
        file_handler = RotatingFileHandler(
            logfile,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if _STREAM_ENABLED:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    for handler in handlers:
        root.addHandler(handler)

    level = os.getenv("LOG_LEVEL") or os.getenv("GLOBAL_LOG_LEVEL") or "INFO"
    try:
        root.setLevel(level.upper())
    except Exception:
        root.setLevel(logging.INFO)

    root._global_log_configured = True  # type: ignore[attr-defined]


_configure_logging()
