"""Strategy shared-state adapter (manager-backed or local)."""
from __future__ import annotations

import os
import threading
from multiprocessing.managers import SyncManager
from typing import Any, Mapping

STATE_KEYS = ("global_strategy", "targets", "cooldowns")


def _default_state() -> dict[str, dict[str, Any]]:
    return {key: {} for key in STATE_KEYS}


class StrategyStateManager(SyncManager):
    """SyncManager for sharing strategy state across processes."""


def _register_server():
    StrategyStateManager.register("get_state", callable=_default_state)


def _register_client():
    StrategyStateManager.register("get_state")


def serve_strategy_state(host: str, port: int, authkey: str) -> None:
    _register_server()
    manager = StrategyStateManager(address=(host, port), authkey=authkey.encode())
    server = manager.get_server()
    server.serve_forever()


def connect_strategy_state(host: str, port: int, authkey: str) -> "StrategyStateAdapter":
    _register_client()
    manager = StrategyStateManager(address=(host, port), authkey=authkey.encode())
    manager.connect()
    state = manager.get_state()
    return StrategyStateAdapter(state=state)


class StrategyStateAdapter:
    """Read/write access to strategy state."""

    def __init__(self, state: Mapping[str, Mapping[str, Any]] | None = None, lock: threading.Lock | None = None) -> None:
        self._state = state if state is not None else _default_state()
        self._lock = lock or threading.Lock()

    def read(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(self._state.get(key, {})) for key in STATE_KEYS}

    def write(self, update: Mapping[str, Mapping[str, Any]] | None) -> None:
        if not update:
            return
        with self._lock:
            for key in STATE_KEYS:
                if key in update:
                    self._state[key] = dict(update.get(key) or {})


def build_strategy_state_adapter() -> StrategyStateAdapter:
    host = os.getenv("STRATEGY_STATE_HOST")
    port = os.getenv("STRATEGY_STATE_PORT")
    if host and port:
        authkey = os.getenv("STRATEGY_STATE_AUTHKEY", "strategy")
        return connect_strategy_state(host, int(port), authkey)
    return StrategyStateAdapter()
