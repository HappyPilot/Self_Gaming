import base64
import logging
import os
import time
import uuid
from typing import Optional

logger = logging.getLogger("frame_transport")

try:
    from multiprocessing import shared_memory
    SHM_AVAILABLE = True
except Exception:
    shared_memory = None
    SHM_AVAILABLE = False

_SHM_LOG_INTERVAL_SEC = 5.0
_last_shm_log = 0.0


def _log_shm_debug(message: str, *args) -> None:
    global _last_shm_log
    now = time.time()
    if now - _last_shm_log >= _SHM_LOG_INTERVAL_SEC:
        logger.debug(message, *args)
        _last_shm_log = now


def get_transport_mode() -> str:
    return os.getenv("FRAME_TRANSPORT", "mqtt").lower()


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class ShmFrameRing:
    def __init__(self, max_bytes: int, slots: int, prefix: str = "sg_frame"):
        if not SHM_AVAILABLE:
            raise RuntimeError("shared_memory not available")
        self.max_bytes = max(1, _safe_int(max_bytes, 1024 * 1024))
        self.slots = max(1, _safe_int(slots, 4))
        self.prefix = prefix or "sg_frame"
        self._segments = []
        self._index = 0
        self._seq = 0
        self._create_segments()

    def _create_segments(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        for idx in range(self.slots):
            name = f"{self.prefix}_{os.getpid()}_{suffix}_{idx}"
            shm = shared_memory.SharedMemory(name=name, create=True, size=self.max_bytes)
            self._segments.append(shm)

    def write(self, data: bytes) -> Optional[dict]:
        if not data or len(data) > self.max_bytes:
            return None
        shm = self._segments[self._index]
        self._index = (self._index + 1) % self.slots
        self._seq += 1
        shm.buf[: len(data)] = data
        return {"shm_name": shm.name, "shm_size": len(data), "shm_seq": self._seq}

    def close(self) -> None:
        for shm in self._segments:
            try:
                shm.close()
            except Exception:
                pass
        for shm in self._segments:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


def read_shm_bytes(name: str, size: int) -> Optional[bytes]:
    if not SHM_AVAILABLE:
        return None
    if not name or not size:
        return None
    size = _safe_int(size, 0)
    if size <= 0:
        return None
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        _log_shm_debug("SHM segment not found: %s", name)
        return None
    except Exception:
        _log_shm_debug("Failed to open shm segment %s", name)
        return None
    try:
        size = min(size, shm.size)
        return bytes(shm.buf[:size])
    finally:
        try:
            shm.close()
        except Exception:
            pass


def get_frame_bytes(payload: dict) -> Optional[bytes]:
    if not isinstance(payload, dict):
        return None
    if payload.get("transport") == "shm":
        return read_shm_bytes(payload.get("shm_name"), payload.get("shm_size"))
    encoded = payload.get("image_b64")
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None


def get_frame_b64(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    encoded = payload.get("image_b64")
    if encoded:
        return encoded
    data = get_frame_bytes(payload)
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")
