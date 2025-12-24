#!/usr/bin/env python3
"""Reflex agent scaffold that reads shared strategy state."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time

from shared_state.strategy_state import build_strategy_state_adapter

logging.basicConfig(level=os.getenv("REFLEX_LOG_LEVEL", "INFO"))
logger = logging.getLogger("reflex_agent")
stop_event = threading.Event()

POLL_SEC = float(os.getenv("REFLEX_POLL_SEC", "0.5"))
LOG_EVERY_SEC = float(os.getenv("REFLEX_LOG_EVERY_SEC", "5.0"))


class ReflexAgent:
    def __init__(self) -> None:
        self.adapter = build_strategy_state_adapter()

    def run(self) -> None:
        last_log = 0.0
        while not stop_event.is_set():
            state = self.adapter.read()
            now = time.time()
            if now - last_log >= LOG_EVERY_SEC:
                logger.info("Strategy state snapshot: %s", state)
                last_log = now
            stop_event.wait(POLL_SEC)


def _handle_signal(signum, _frame):
    logger.info("Signal %s received", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    ReflexAgent().run()


if __name__ == "__main__":
    main()
