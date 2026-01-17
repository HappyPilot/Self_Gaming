from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import base64
import io
import pickle

import numpy as np
from PIL import Image
import zmq


@dataclass(frozen=True)
class NitroGenConfig:
    host: str
    port: int
    timeout_s: float = 5.0


class NitroGenClient:
    """Thin ZMQ client for the NitroGen inference server (scripts/serve.py)."""

    def __init__(self, cfg: NitroGenConfig) -> None:
        self.cfg = cfg
        self.context = zmq.Context()
        self.socket = self._connect()

    def _connect(self) -> zmq.Socket:
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        timeout_ms = max(int(self.cfg.timeout_s * 1000), 100)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.connect(f"tcp://{self.cfg.host}:{self.cfg.port}")
        return socket

    def _reset_socket(self) -> None:
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        self.socket = self._connect()

    def infer(self, jpeg_b64: str, intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        del intent  # NitroGen serve.py does not accept intents yet.
        img_bytes = base64.b64decode(jpeg_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        request = {"type": "predict", "image": np.asarray(img)}
        try:
            self.socket.send(pickle.dumps(request))
            raw = self.socket.recv()
        except zmq.error.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"NitroGen inference timed out after {self.cfg.timeout_s}s") from exc
        except Exception:
            self._reset_socket()
            raise
        response = pickle.loads(raw)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "NitroGen error"))
        return response.get("pred", {})

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.context.term()
