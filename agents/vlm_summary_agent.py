#!/usr/bin/env python3
"""VLM summary agent: turns frames into structured scene summaries."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import requests
except ImportError:
    requests = None

import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_b64

logging.basicConfig(level=os.getenv("VLM_SUMMARY_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm_summary_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
SCENE_SUMMARY_TOPIC = os.getenv("SCENE_SUMMARY_TOPIC", "scene/summary")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")

VLM_ENDPOINT = os.getenv("VLM_ENDPOINT") or os.getenv("TEACHER_LOCAL_ENDPOINT") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1/chat/completions"
VLM_MODEL = os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL") or "llama"
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", ""))
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "20"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))

SUMMARY_INTERVAL_SEC = float(os.getenv("VLM_SUMMARY_INTERVAL_SEC", "2.5"))
STALE_PUBLISH_SEC = float(os.getenv("VLM_SUMMARY_STALE_PUBLISH_SEC", "10"))

LOG_DIR = Path(os.getenv("VLM_SUMMARY_LOG_DIR", "/mnt/ssd/logs/vlm_summary"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "events.jsonl"

stop_event = threading.Event()


def normalize_summary(payload: object, now: Optional[float] = None) -> Dict[str, object]:
    now = now or time.time()
    if not isinstance(payload, dict):
        payload = {"summary": str(payload)}
    game = payload.get("game") or payload.get("game_id") or "unknown_game"
    risk = str(payload.get("risk") or "unknown").strip().lower() or "unknown"
    if risk not in {"low", "medium", "high", "unknown"}:
        risk = "unknown"
    def _as_text(value) -> str:
        if value is None:
            return ""
        return str(value)
    return {
        "game": _as_text(game),
        "summary": _as_text(payload.get("summary")),
        "player_state": _as_text(payload.get("player_state")),
        "enemies": _as_text(payload.get("enemies")),
        "objectives": _as_text(payload.get("objectives")),
        "ui": _as_text(payload.get("ui")),
        "risk": risk,
        "recommended_intent": _as_text(payload.get("recommended_intent")),
        "timestamp": float(payload.get("timestamp") or now),
    }


def _hash_summary(summary: Dict[str, object]) -> str:
    try:
        blob = json.dumps(summary, sort_keys=True, ensure_ascii=True)
    except Exception:
        blob = str(summary)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def _build_messages(image_b64: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a vision analyst for video games. "
                "Return ONLY valid JSON with keys: game, summary, player_state, enemies, objectives, ui, risk, recommended_intent. "
                "risk must be one of: low, medium, high. Keep summary concise."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the scene and return JSON only."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        },
    ]


class VlmSummaryAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="vlm_summary", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self._lock = threading.Lock()
        self._latest_frame: Optional[dict] = None
        self._latest_frame_ts: float = 0.0
        self._last_request: float = 0.0
        self._last_publish: float = 0.0
        self._last_summary: Optional[Dict[str, object]] = None
        self._last_summary_ts: float = 0.0

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(FRAME_TOPIC, 0)])
            logger.info("VLM summary agent connected: %s", FRAME_TOPIC)
        else:
            logger.error("VLM summary agent failed to connect rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        if msg.topic != FRAME_TOPIC:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._latest_frame = payload
            self._latest_frame_ts = time.time()

    def _log_event(self, status: str, latency_ms: Optional[float], summary: Optional[Dict[str, object]] = None, error: Optional[str] = None):
        record = {
            "ts": time.time(),
            "status": status,
            "latency_ms": latency_ms,
        }
        if summary:
            record["game"] = summary.get("game")
            record["summary_hash"] = _hash_summary(summary)
        if error:
            record["error"] = error
        try:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _publish_summary(self, summary: Dict[str, object], status: str, latency_ms: Optional[float] = None, error: Optional[str] = None):
        now = time.time()
        payload = {
            "ok": status == "ok",
            "status": status,
            "timestamp": now,
            "summary": summary,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error:
            payload["error"] = error
        self.client.publish(SCENE_SUMMARY_TOPIC, json.dumps(payload))
        self._last_publish = now
        if MEM_STORE_TOPIC:
            scope = summary.get("game") or "unknown_game"
            store_payload = {
                "op": "set",
                "key": f"scene_summary:{scope}",
                "value": summary,
                "timestamp": now,
            }
            self.client.publish(MEM_STORE_TOPIC, json.dumps(store_payload))

    def _request_summary(self, image_b64: str) -> Dict[str, object]:
        if requests is None:
            raise RuntimeError("requests package is unavailable")
        headers = {"Content-Type": "application/json"}
        if VLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLM_API_KEY}"
        payload = {
            "model": VLM_MODEL,
            "messages": _build_messages(image_b64),
            "temperature": 0.2,
            "max_tokens": max(64, VLM_MAX_TOKENS),
            "stream": False,
        }
        response = requests.post(VLM_ENDPOINT, headers=headers, json=payload, timeout=VLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                return json.loads(content) if isinstance(content, str) else content
            if "message" in data:
                return data["message"]
        raise ValueError(f"Unexpected response payload: {data}")

    def run(self) -> None:
        self.client.connect(MQTT_HOST, MQTT_PORT, 60)
        self.client.loop_start()
        while not stop_event.is_set():
            now = time.time()
            if now - self._last_request < SUMMARY_INTERVAL_SEC:
                stop_event.wait(0.1)
                continue
            with self._lock:
                frame_payload = self._latest_frame
            if not frame_payload:
                stop_event.wait(0.1)
                continue
            image_b64 = get_frame_b64(frame_payload)
            if not image_b64:
                stop_event.wait(0.1)
                continue
            self._last_request = now
            try:
                start = time.time()
                raw_summary = self._request_summary(image_b64)
                latency_ms = (time.time() - start) * 1000.0
                summary = normalize_summary(raw_summary)
                self._last_summary = summary
                self._last_summary_ts = time.time()
                self._publish_summary(summary, status="ok", latency_ms=latency_ms)
                self._log_event("ok", latency_ms, summary=summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM summary failed: %s", exc)
                self._log_event("error", None, summary=self._last_summary, error=str(exc))
                if self._last_summary and now - self._last_publish >= STALE_PUBLISH_SEC:
                    stale_summary = dict(self._last_summary)
                    stale_summary["stale_age_sec"] = max(0.0, now - self._last_summary_ts)
                    self._publish_summary(stale_summary, status="stale", error=str(exc))
            stop_event.wait(0.05)
        self.client.loop_stop()
        self.client.disconnect()


def _handle_signal(signum, _frame):
    logger.info("Signal %s received", signum)
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    VlmSummaryAgent().run()


if __name__ == "__main__":
    main()
