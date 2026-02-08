#!/usr/bin/env python3
"""Publish SigLIP prompt similarity scores from incoming frames."""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import signal
import threading
import time
from typing import List, Optional

import paho.mqtt.client as mqtt
from PIL import Image

from utils.frame_transport import get_frame_bytes

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor
except Exception:  # noqa: BLE001
    torch = None
    F = None
    SiglipModel = None
    SiglipProcessor = None
    AutoModel = None
    AutoProcessor = None

logging.basicConfig(level=os.getenv("SIGLIP_PROMPT_LOG_LEVEL", "INFO"), format="[siglip_prompt] %(message)s")
logger = logging.getLogger("siglip_prompt_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
PROMPT_TOPIC = os.getenv("VISION_PROMPT_TOPIC", "vision/prompt_scores")
PROMPT_CONFIG_TOPIC = os.getenv("VISION_PROMPT_CONFIG_TOPIC", "")

MODEL_ID = os.getenv("SIGLIP_PROMPT_MODEL_ID", "google/siglip2-base-patch16-224").strip()
PROMPTS_RAW = os.getenv(
    "SIGLIP_PROMPTS",
    "enemy,boss,npc,player,loot,chest,portal,waypoint,minimap,inventory,quest,dialog,health bar,mana bar,death screen,loading screen,pause menu",
)
PROMPTS = [prompt.strip() for prompt in PROMPTS_RAW.split(",") if prompt.strip()]

DEVICE_RAW = os.getenv("SIGLIP_PROMPT_DEVICE", "auto").strip().lower()
FP16 = os.getenv("SIGLIP_PROMPT_FP16", "0") not in {"0", "false", "False"}
INTERVAL_SEC = float(os.getenv("SIGLIP_PROMPT_INTERVAL_SEC", "2.0"))
MAX_SIZE = int(os.getenv("SIGLIP_PROMPT_MAX_SIZE", "512"))
TOP_K = int(os.getenv("SIGLIP_PROMPT_TOP_K", "5"))
MIN_SCORE = float(os.getenv("SIGLIP_PROMPT_MIN_SCORE", "0.2"))
QUEUE_MAX = int(os.getenv("SIGLIP_PROMPT_QUEUE", "2"))
FRAME_TIMEOUT = float(os.getenv("SIGLIP_PROMPT_FRAME_TIMEOUT", "5.0"))
PROMPT_MIN_LEN = int(os.getenv("SIGLIP_PROMPT_MIN_LEN", "2"))
PROMPT_MAX = int(os.getenv("SIGLIP_PROMPT_MAX", "24"))
PROMPT_CONFIG_INTERVAL_SEC = float(os.getenv("SIGLIP_PROMPT_CONFIG_INTERVAL_SEC", "5.0"))

stop_event = threading.Event()
frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max(1, QUEUE_MAX))
prompt_lock = threading.Lock()
pending_prompts: Optional[List[str]] = None
last_prompt_config_ts = 0.0


def _resolve_device() -> str:
    if DEVICE_RAW and DEVICE_RAW not in {"auto", "default"}:
        return DEVICE_RAW
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SiglipPromptScorer:
    def __init__(self, model_id: str, prompts: List[str], device: str, fp16: bool) -> None:
        if torch is None or F is None or (SiglipModel is None and AutoModel is None):
            raise RuntimeError("SigLIP dependencies are unavailable")
        if not prompts:
            raise ValueError("SIGLIP_PROMPTS must contain at least one prompt")
        self.prompts = prompts
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        if SiglipProcessor is not None:
            self.processor = SiglipProcessor.from_pretrained(model_id)
        elif AutoProcessor is not None:
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise RuntimeError("No compatible processor found for SigLIP")
        if SiglipModel is not None:
            self.model = SiglipModel.from_pretrained(model_id, torch_dtype=torch.float16 if self.fp16 else None)
        else:
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if self.fp16 else None)
        self.model.eval()
        self.model.to(device)
        if not hasattr(self.model, "get_text_features") or not hasattr(self.model, "get_image_features"):
            raise RuntimeError("SigLIP model is missing text/image feature helpers")
        self.text_features = self._encode_text()

    def _encode_text(self) -> "torch.Tensor":
        inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def update_prompts(self, prompts: List[str]) -> None:
        if not prompts:
            return
        self.prompts = prompts
        self.text_features = self._encode_text()

    def score(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.fp16:
            inputs = {key: value.half() for key, value in inputs.items()}
        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        scores = (image_features @ self.text_features.T).squeeze(0)
        scores = scores.float().cpu().numpy().tolist()
        return {prompt: float(score) for prompt, score in zip(self.prompts, scores)}


def _decode_frame(raw: bytes) -> Optional[Image.Image]:
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None
    if MAX_SIZE > 0:
        width, height = image.size
        longer = max(width, height)
        if longer > MAX_SIZE:
            scale = MAX_SIZE / float(longer)
            image = image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    return image


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        topics = [(FRAME_TOPIC, 0)]
        if PROMPT_CONFIG_TOPIC:
            topics.append((PROMPT_CONFIG_TOPIC, 0))
        client.subscribe(topics)
        logger.info("ready: topics=%s prompts=%d device=%s", [t for t, _ in topics], len(PROMPTS), _resolve_device())
    else:
        logger.error("mqtt connect failed rc=%s", rc)

def _normalize_prompt_list(values: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for value in values:
        token = str(value).strip().lower()
        if not token or len(token) < PROMPT_MIN_LEN:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if PROMPT_MAX and len(cleaned) >= PROMPT_MAX:
            break
    return cleaned


def _extract_prompt_config(payload: dict) -> Optional[List[str]]:
    if not isinstance(payload, dict):
        return None
    prompts = payload.get("prompts")
    if not prompts:
        prompts = []
        for key in ("object_prompts", "ui_prompts"):
            items = payload.get(key) or []
            if isinstance(items, list):
                prompts.extend(items)
    if not isinstance(prompts, list) or not prompts:
        return None
    return _normalize_prompt_list(prompts)


def _on_message(_client, _userdata, msg):
    global pending_prompts, last_prompt_config_ts
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return
    if PROMPT_CONFIG_TOPIC and msg.topic == PROMPT_CONFIG_TOPIC:
        now = time.time()
        if PROMPT_CONFIG_INTERVAL_SEC > 0 and (now - last_prompt_config_ts) < PROMPT_CONFIG_INTERVAL_SEC:
            return
        prompts = _extract_prompt_config(payload)
        if not prompts:
            return
        with prompt_lock:
            pending_prompts = prompts
            last_prompt_config_ts = now
        logger.info("received prompt config prompts=%d", len(prompts))
        return
    raw = get_frame_bytes(payload)
    if not raw:
        return
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        frame_queue.put_nowait(raw)
    except queue.Full:
        pass


def _format_top(scores: dict) -> List[dict]:
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top = []
    for label, score in items:
        if score < MIN_SCORE:
            continue
        top.append({"label": label, "score": round(score, 4)})
        if len(top) >= TOP_K:
            break
    return top


def run() -> None:
    global pending_prompts
    device = _resolve_device()
    scorer = SiglipPromptScorer(MODEL_ID, PROMPTS, device, FP16)
    current_prompts = list(PROMPTS)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="siglip_prompt")
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    last_infer = 0.0
    last_frame_ts = time.time()
    while not stop_event.is_set():
        try:
            raw = frame_queue.get(timeout=0.5)
        except queue.Empty:
            if time.time() - last_frame_ts > FRAME_TIMEOUT:
                time.sleep(0.1)
            continue
        now = time.time()
        if (now - last_infer) < INTERVAL_SEC:
            continue
        last_frame_ts = now
        with prompt_lock:
            if pending_prompts and pending_prompts != current_prompts:
                scorer.update_prompts(pending_prompts)
                current_prompts = list(pending_prompts)
                pending_prompts = None
                logger.info("updated prompt set size=%d", len(current_prompts))
        image = _decode_frame(raw)
        if image is None:
            continue
        scores = scorer.score(image)
        payload = {
            "ok": True,
            "timestamp": time.time(),
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "top": _format_top(scores),
            "model_id": MODEL_ID,
            "backend": "siglip_prompt",
        }
        client.publish(PROMPT_TOPIC, json.dumps(payload), qos=0)
        last_infer = time.time()

    client.loop_stop()
    client.disconnect()


def _handle_signal(_signum, _frame):
    stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("siglip_prompt crashed: %s", exc)
