"""Shared LLM gate to avoid concurrent requests across agents."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Optional

GATE_ENABLED = os.getenv("LLM_GATE_ENABLED", "1") != "0"
GATE_FILE_DEFAULT = "/mnt/ssd/logs/llm_gate.lock"
PAUSE_FILE_DEFAULT = "/mnt/ssd/logs/llm_pause"
GATE_TTL_S = float(os.getenv("LLM_GATE_TTL_S", "120"))
GATE_WAIT_S = float(os.getenv("LLM_GATE_WAIT_S", "30"))
GATE_POLL_SEC = float(os.getenv("LLM_GATE_POLL_SEC", "0.2"))
SKIP_IF_GATED = os.getenv("LLM_SKIP_IF_GATED", "1") != "0"


def _path_from_env(env_key: str, default: str) -> Optional[Path]:
    value = os.getenv(env_key, default).strip()
    if not value:
        return None
    return Path(value)


def _path_available(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        return path.parent.exists()
    except OSError:
        return False


def _is_stale(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return True
    return age > GATE_TTL_S


def gate_active() -> bool:
    if not GATE_ENABLED:
        return False
    path = _path_from_env("LLM_GATE_FILE", GATE_FILE_DEFAULT)
    if not _path_available(path):
        return False
    assert path is not None
    if not path.exists():
        return False
    if _is_stale(path):
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def pause_active() -> bool:
    path = _path_from_env("LLM_PAUSE_FILE", PAUSE_FILE_DEFAULT)
    if not _path_available(path):
        return False
    assert path is not None
    return path.exists()


def blocked_reason() -> str:
    if pause_active():
        return "pause_file"
    if gate_active() and SKIP_IF_GATED:
        return "gate_active"
    return ""


def acquire_gate(purpose: str, wait_s: Optional[float] = None) -> bool:
    if not GATE_ENABLED:
        return True
    path = _path_from_env("LLM_GATE_FILE", GATE_FILE_DEFAULT)
    if not _path_available(path):
        return True
    assert path is not None
    wait_s = GATE_WAIT_S if wait_s is None else float(wait_s)
    deadline = time.time() + max(0.0, wait_s)
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "purpose": purpose,
        "ts": time.time(),
    }
    while True:
        if path.exists() and _is_stale(path):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                return False
            time.sleep(GATE_POLL_SEC)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload))
        return True


def release_gate() -> None:
    if not GATE_ENABLED:
        return
    path = _path_from_env("LLM_GATE_FILE", GATE_FILE_DEFAULT)
    if not _path_available(path):
        return
    assert path is not None
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
