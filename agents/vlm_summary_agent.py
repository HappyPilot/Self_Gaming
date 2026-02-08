#!/usr/bin/env python3
"""VLM summary agent: turns frames into structured scene summaries."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import io
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

try:
    from PIL import Image
except ImportError:
    Image = None

import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_b64, get_frame_bytes

logging.basicConfig(level=os.getenv("VLM_SUMMARY_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm_summary_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
SCENE_SUMMARY_TOPIC = os.getenv("SCENE_SUMMARY_TOPIC", "scene/summary")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")

VLM_ENDPOINT = os.getenv("VLM_ENDPOINT") or os.getenv("TEACHER_LOCAL_ENDPOINT") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1/chat/completions"
VLM_MODEL = os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL") or "llama"
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", ""))
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "20"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))
VLM_DISABLE_IMAGE = os.getenv("VLM_DISABLE_IMAGE", "0") != "0"
VLM_IMAGE_MAX_DIM = int(os.getenv("VLM_IMAGE_MAX_DIM", "320"))
VLM_IMAGE_FORMAT = os.getenv("VLM_IMAGE_FORMAT", "PNG")
VLM_IMAGE_JPEG_QUALITY = int(os.getenv("VLM_IMAGE_JPEG_QUALITY", "85"))

SUMMARY_INTERVAL_SEC = float(os.getenv("VLM_SUMMARY_INTERVAL_SEC", "2.5"))
STALE_PUBLISH_SEC = float(os.getenv("VLM_SUMMARY_STALE_PUBLISH_SEC", "10"))

VLM_REQUIRE_GAME = os.getenv("VLM_REQUIRE_GAME", "1") != "0"
VLM_GAME_EXTRA_SAMPLES = int(os.getenv("VLM_GAME_EXTRA_SAMPLES", "3"))
VLM_GAME_RETRY_INTERVAL_SEC = float(os.getenv("VLM_GAME_RETRY_INTERVAL_SEC", "45"))

LOG_DIR = Path(os.getenv("VLM_SUMMARY_LOG_DIR", "/mnt/ssd/logs/vlm_summary"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "events.jsonl"

stop_event = threading.Event()



def _is_unknown_game(value: object) -> bool:
    if value is None:
        return True
    cleaned = str(value).strip().lower()
    return cleaned in {"", "unknown", "unknown_game", "none", "n/a", "null"}


def _select_majority_game(votes: list[str]) -> str | None:
    if not votes:
        return None
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


class GameDetector:
    def __init__(self, extra_samples: int = 3) -> None:
        self.extra_samples = max(0, int(extra_samples))
        self.attempts = 0
        self.target_attempts = 1
        self.votes: list[str] = []
        self.detected_game: str | None = None

    def observe(self, game_guess: object) -> str | None:
        self.attempts += 1
        if not _is_unknown_game(game_guess):
            self.votes.append(str(game_guess).strip())
        if self.attempts == 1:
            if self.votes:
                self.detected_game = self.votes[0]
                return self.detected_game
            self.target_attempts = 1 + self.extra_samples
        if self.attempts >= self.target_attempts:
            self.detected_game = _select_majority_game(self.votes) or "unknown_game"
            return self.detected_game
        return None




def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
        return stripped.strip("`")
    return stripped


def _extract_json_block(text: str) -> str:
    if not text:
        return ""
    stripped = _strip_code_fences(text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


def _parse_json_payload(raw: str):
    if not raw:
        return None
    block = _extract_json_block(raw)
    try:
        parsed = json.loads(block)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

def normalize_summary(payload: object, now: Optional[float] = None) -> Dict[str, object]:
    now = now or time.time()
    if not isinstance(payload, dict):
        parsed = _parse_json_payload(str(payload))
        payload = parsed if isinstance(parsed, dict) else {"summary": str(payload)}
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


def prepare_image_b64(payload: dict) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None
    data = get_frame_bytes(payload)
    if not data:
        raw = get_frame_b64(payload)
        return (raw, "image/jpeg") if raw else (None, None)
    if Image is None:
        return base64.b64encode(data).decode("ascii"), "image/jpeg"
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        max_dim = VLM_IMAGE_MAX_DIM
        if max_dim and max_dim > 0:
            img.thumbnail((max_dim, max_dim))
        fmt = str(VLM_IMAGE_FORMAT or "PNG").upper()
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=VLM_IMAGE_JPEG_QUALITY)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"
        return base64.b64encode(buf.getvalue()).decode("ascii"), mime
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to preprocess image: %s", exc)
        raw = get_frame_b64(payload)
        return (raw, "image/jpeg") if raw else (None, None)


def build_messages(image_b64: Optional[str], image_mime: Optional[str], scene_text: str, object_summary: str) -> list:
    context_bits = []
    if scene_text:
        context_bits.append(f"Scene text:\\n{scene_text}")
    if object_summary:
        context_bits.append(f"Objects:\\n{object_summary}")
    context = "\\n\\n".join(context_bits)
    system_msg = (
        "You are a vision analyst for video games. First, identify the game in the image. If you are not confident, set game to unknown_game and do not guess. "
        "Return ONLY valid JSON with keys: game, summary, player_state, enemies, objectives, ui, risk, recommended_intent. "
        "risk must be one of: low, medium, high. Keep summary concise."
    )
    if image_b64:
        mime = image_mime or "image/jpeg"
        return [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze the scene and return JSON only.\\n\\n{context}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            },
        ]
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Analyze the scene using the text context below and return JSON only.\\n\\n{context}"},
    ]


class VlmSummaryAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="vlm_summary")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self._lock = threading.Lock()
        self._latest_frame: Optional[dict] = None
        self._latest_frame_ts: float = 0.0
        self._latest_scene_text: str = ""
        self._latest_object_summary: str = ""
        self._last_request: float = 0.0
        self._last_publish: float = 0.0
        self._last_summary: Optional[Dict[str, object]] = None
        self._last_summary_ts: float = 0.0
        self._game_detector = GameDetector(extra_samples=VLM_GAME_EXTRA_SAMPLES)
        self._next_game_detect_ts: float = 0.0

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(FRAME_TOPIC, 0), (SCENE_TOPIC, 0)])
            logger.info("VLM summary agent connected: %s, %s", FRAME_TOPIC, SCENE_TOPIC)
        else:
            logger.error("VLM summary agent failed to connect rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        if msg.topic == FRAME_TOPIC:
            with self._lock:
                self._latest_frame = payload
                self._latest_frame_ts = time.time()
            return
        if msg.topic == SCENE_TOPIC:
            text_entries = payload.get("text") or []
            if isinstance(text_entries, list):
                scene_text = " ".join(str(item) for item in text_entries)
            else:
                scene_text = str(text_entries)
            objects = payload.get("objects") or payload.get("enemies") or []
            object_labels = []
            if isinstance(objects, list):
                for obj in objects[:12]:
                    if isinstance(obj, dict):
                        label = obj.get("label") or obj.get("class") or obj.get("name")
                        if label:
                            object_labels.append(str(label))
            with self._lock:
                self._latest_scene_text = scene_text.strip()
                self._latest_object_summary = ", ".join(object_labels)

    def _log_event(self, status: str, latency_ms: Optional[float], summary: Optional[Dict[str, object]] = None, error: Optional[str] = None, source: Optional[str] = None):
        record = {
            "ts": time.time(),
            "status": status,
            "latency_ms": latency_ms,
        }
        if summary:
            record["game"] = summary.get("game")
            record["summary_hash"] = _hash_summary(summary)
        if source:
            record["source"] = source
        if error:
            record["error"] = error
        try:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _publish_summary(self, summary: Dict[str, object], status: str, latency_ms: Optional[float] = None, error: Optional[str] = None, source: Optional[str] = None):
        now = time.time()
        payload = {
            "ok": status == "ok",
            "status": status,
            "timestamp": now,
            "summary": summary,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if source:
            payload["source"] = source
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

    def _request_summary(self, messages: list) -> Dict[str, object]:
        if requests is None:
            raise RuntimeError("requests package is unavailable")
        headers = {"Content-Type": "application/json"}
        if VLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLM_API_KEY}"
        payload = {
            "model": VLM_MODEL,
            "messages": messages,
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
                if isinstance(content, str):
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        cleaned = "\n".join(
                            line for line in cleaned.splitlines() if not line.strip().startswith("```")
                        ).strip()
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return {"summary": content}
                return content
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
                scene_text = self._latest_scene_text
                object_summary = self._latest_object_summary
            if not frame_payload:
                stop_event.wait(0.1)
                continue
            image_b64 = None
            image_mime = None
            if not VLM_DISABLE_IMAGE:
                image_b64, image_mime = prepare_image_b64(frame_payload)
            if not image_b64 and not scene_text:
                stop_event.wait(0.1)
                continue
            if VLM_REQUIRE_GAME and self._game_detector.detected_game is None and now < self._next_game_detect_ts:
                stop_event.wait(0.1)
                continue
            self._last_request = now
            try:
                start = time.time()
                messages = build_messages(image_b64, image_mime, scene_text, object_summary)
                source = "vlm" if image_b64 else "text"
                raw_summary = self._request_summary(messages)
                latency_ms = (time.time() - start) * 1000.0
                summary = normalize_summary(raw_summary)
                self._last_summary = summary
                self._last_summary_ts = time.time()
                if VLM_REQUIRE_GAME and self._game_detector.detected_game is None:
                    detected_game = self._game_detector.observe(summary.get("game"))
                    if detected_game is None:
                        self._next_game_detect_ts = now + VLM_GAME_RETRY_INTERVAL_SEC
                        self._log_event("pending_game", latency_ms, summary=summary, source=source)
                        stop_event.wait(0.05)
                        continue
                    logger.info("VLM game detected: %s (attempts=%s)", detected_game, self._game_detector.attempts)
                    if _is_unknown_game(summary.get("game")):
                        summary["game"] = detected_game
                self._publish_summary(summary, status="ok", latency_ms=latency_ms, source=source)
                self._log_event("ok", latency_ms, summary=summary, source=source)
            except Exception as exc:  # noqa: BLE001
                if image_b64 and scene_text:
                    try:
                        start = time.time()
                        messages = build_messages(None, None, scene_text, object_summary)
                        raw_summary = self._request_summary(messages)
                        latency_ms = (time.time() - start) * 1000.0
                        summary = normalize_summary(raw_summary)
                        self._last_summary = summary
                        self._last_summary_ts = time.time()
                        self._publish_summary(summary, status="ok", latency_ms=latency_ms, source="text")
                        self._log_event("ok", latency_ms, summary=summary, source="text")
                        continue
                    except Exception as fallback_exc:  # noqa: BLE001
                        logger.warning("VLM summary failed: %s", fallback_exc)
                        self._log_event("error", None, summary=self._last_summary, error=str(fallback_exc))
                else:
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
