#!/usr/bin/env python3
"""Tactical agent scaffold that writes shared strategy state."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time

from shared_state.strategy_state import build_strategy_state_adapter

logging.basicConfig(level=os.getenv("TACTICAL_LOG_LEVEL", "INFO"))
logger = logging.getLogger("tactical_agent")
stop_event = threading.Event()

UPDATE_SEC = float(os.getenv("TACTICAL_UPDATE_SEC", "2.0"))
STRATEGY_MODE = os.getenv("TACTICAL_STRATEGY_MODE", "scan")


class TacticalAgent:
    def __init__(self) -> None:
        self.adapter = build_strategy_state_adapter()

    def run(self) -> None:
        while not stop_event.is_set():
            update = {
                "global_strategy": {"mode": STRATEGY_MODE, "ts": time.time()},
                "targets": {},
                "cooldowns": {},
            }
            self.adapter.write(update)
            logger.debug("Strategy state updated: %s", update)
            stop_event.wait(UPDATE_SEC)


def _handle_signal(signum, _frame):
    logger.info("Signal %s received", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    TacticalAgent().run()


if __name__ == "__main__":
    main()
