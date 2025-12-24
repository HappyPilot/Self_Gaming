"""Strategy shared-state manager server."""
from __future__ import annotations

import os

from shared_state.strategy_state import serve_strategy_state


def main() -> None:
    host = os.getenv("STRATEGY_STATE_HOST", "0.0.0.0")
    port = int(os.getenv("STRATEGY_STATE_PORT", "54001"))
    authkey = os.getenv("STRATEGY_STATE_AUTHKEY", "strategy")
    serve_strategy_state(host, port, authkey)


if __name__ == "__main__":
    main()
