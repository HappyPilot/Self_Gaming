from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests


@dataclass(frozen=True)
class NitroGenConfig:
    base_url: str
    timeout_s: float = 2.0


class NitroGenClient:
    """Thin HTTP client for the NitroGen inference server.

    NOTE: The exact endpoint path/payload depends on scripts/serve.py.
    Start the server and check its logs/docs; then adjust `infer()`.
    """

    def __init__(self, cfg: NitroGenConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()

    def infer(self, jpeg_b64: str, intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.cfg.base_url.rstrip('/')}/infer"  # <-- change if serve.py uses a different route
        payload = {
            "image_b64_jpeg": jpeg_b64,
            "intent": intent or {},
        }
        r = self.session.post(url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()
