#!/usr/bin/env python3
"""
Sparse anchor detector (OWL-ViT).
Subscribes to vision/frame/preview + vision/embeddings, publishes abstract anchors to vision/anchors.
"""
from __future__ import annotations

import io
import json
import logging
import os
import signal
import threading
import time
from typing import Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None
try:
    from transformers import OwlViTForObjectDetection, OwlViTProcessor
except Exception:  # noqa: BLE001
    OwlViTForObjectDetection = None
    OwlViTProcessor = None


logging.basicConfig(level=os.getenv("ANCHOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("anchor_detector")
stop_event = threading.Event()

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
EMBED_TOPIC = os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings")
ANCHOR_TOPIC = os.getenv("VISION_ANCHOR_TOPIC", "vision/anchors")

MODEL_ID = os.getenv("ANCHOR_MODEL_ID", "google/owlvit-base-patch32")
PROMPTS_GOAL = [p.strip() for p in os.getenv("ANCHOR_PROMPTS_GOAL", "door,exit,waypoint,objective,portal").split(",") if p.strip()]
PROMPTS_AFFORD = [p.strip() for p in os.getenv("ANCHOR_PROMPTS_AFFORD", "button,lever,handle,interactable,loot,chest").split(",") if p.strip()]
PROMPTS_OBSTACLE = [p.strip() for p in os.getenv("ANCHOR_PROMPTS_OBSTACLE", "wall,barrier,obstacle").split(",") if p.strip()]

CONF_THRESHOLD = float(os.getenv("ANCHOR_CONF_THRESHOLD", "0.2"))
MAX_PER_LABEL = int(os.getenv("ANCHOR_MAX_PER_LABEL", "2"))
MAX_SIZE = int(os.getenv("ANCHOR_MAX_SIZE", "320"))
FRAME_TIMEOUT = float(os.getenv("ANCHOR_FRAME_TIMEOUT", "5.0"))
TORCH_THREADS = int(os.getenv("ANCHOR_THREADS", "2"))
ANCHOR_DEVICE = os.getenv("ANCHOR_DEVICE", "auto").strip().lower()

TRIGGER_TIMER_SEC = float(os.getenv("ANCHOR_TRIGGER_TIMER_SEC", "8.0"))
TRIGGER_STAGNATION_SEC = float(os.getenv("ANCHOR_TRIGGER_STAGNATION_SEC", "4.0"))
TRIGGER_EMBED_SAMPLE_SEC = float(os.getenv("ANCHOR_TRIGGER_EMBED_SAMPLE_SEC", "1.0"))
TRIGGER_EMBED_DELTA_MAX = float(os.getenv("ANCHOR_TRIGGER_EMBED_DELTA_MAX", "0.02"))


def _resolve_device() -> str:
    if ANCHOR_DEVICE and ANCHOR_DEVICE not in {"auto", "default"}:
        return ANCHOR_DEVICE
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_post_threshold() -> float:
    raw = os.getenv("ANCHOR_POST_THRESHOLD", "0.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value)


class AnchorTrigger:
    def __init__(self, timer_sec: float, stagnation_sec: float) -> None:
        self.timer_sec = max(0.1, float(timer_sec))
        self.stagnation_sec = max(0.1, float(stagnation_sec))
        self.last_run = 0.0
        self.stagnant = False

    def mark_stagnant(self, _now: Optional[float] = None) -> None:
        self.stagnant = True

    def clear_stagnant(self) -> None:
        self.stagnant = False

    def should_run(self, now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        if self.last_run <= 0:
            self.last_run = ts
            return True
        if self.stagnant and (ts - self.last_run) >= self.stagnation_sec:
            self.last_run = ts
            return True
        if (ts - self.last_run) >= self.timer_sec:
            self.last_run = ts
            return True
        return False


class AnchorDetector:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="anchor_detector", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.last_frame_bytes: Optional[bytes] = None
        self.last_frame_ts = 0.0
        self.last_embed_ts = 0.0
        self.last_embedding: Optional[List[float]] = None
        self.trigger = AnchorTrigger(TRIGGER_TIMER_SEC, TRIGGER_STAGNATION_SEC)

        self.prompt_entries: List[Tuple[str, str]] = []
        for prompt in PROMPTS_GOAL:
            self.prompt_entries.append((prompt, "goal/salient"))
        for prompt in PROMPTS_AFFORD:
            self.prompt_entries.append((prompt, "affordance/interactive"))
        for prompt in PROMPTS_OBSTACLE:
            self.prompt_entries.append((prompt, "obstacle/blocker"))
        if not self.prompt_entries:
            self.prompt_entries.append(("object", "goal/salient"))
        self.device = _resolve_device()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = [(FRAME_TOPIC, 0)]
            if EMBED_TOPIC:
                topics.append((EMBED_TOPIC, 0))
            client.subscribe(topics)
            logger.info("anchor_detector ready (device=%s prompts=%d)", self.device, len(self.prompt_entries))
        else:
            logger.error("MQTT connect failed: rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == FRAME_TOPIC:
            try:
                payload = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            raw = get_frame_bytes(payload)
            if not raw:
                return
            self.last_frame_bytes = raw
            self.last_frame_ts = time.time()
            return
        if msg.topic == EMBED_TOPIC:
            now = time.time()
            if (now - self.last_embed_ts) < TRIGGER_EMBED_SAMPLE_SEC:
                return
            self.last_embed_ts = now
            try:
                payload = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            embedding = payload.get("embedding") or payload.get("embeddings")
            if not isinstance(embedding, list) or not embedding:
                return
            delta = self._embedding_delta(embedding)
            if delta is None:
                return
            if delta <= TRIGGER_EMBED_DELTA_MAX:
                self.trigger.mark_stagnant(now)
            else:
                self.trigger.clear_stagnant()

    def _embedding_delta(self, embedding: List[float]) -> Optional[float]:
        try:
            vec = [float(v) for v in embedding]
        except (TypeError, ValueError):
            return None
        if not self.last_embedding:
            self.last_embedding = vec
            return None
        if len(vec) != len(self.last_embedding):
            self.last_embedding = vec
            return None
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec, self.last_embedding):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0 or norm_b <= 0:
            self.last_embedding = vec
            return None
        cos = dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
        self.last_embedding = vec
        return max(0.0, 1.0 - cos)

    def _decode_frame(self, raw: bytes):
        try:
            from PIL import Image

            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _resize(self, img):
        if MAX_SIZE <= 0:
            return img
        w, h = img.size
        longer = max(w, h)
        if longer <= MAX_SIZE:
            return img
        scale = MAX_SIZE / float(longer)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size)

    def _publish(self, anchors: List[Dict[str, object]], frame_ts: float) -> None:
        payload = {
            "ok": True,
            "timestamp": time.time(),
            "frame_ts": frame_ts,
            "anchors": anchors,
            "backend": "anchor_owlvit",
            "model_id": MODEL_ID,
        }
        self.client.publish(ANCHOR_TOPIC, json.dumps(payload), qos=0)

    def run(self) -> None:
        if torch is None or OwlViTForObjectDetection is None or OwlViTProcessor is None:
            logger.error("anchor_detector missing torch/transformers dependencies")
            return
        torch.set_num_threads(max(1, TORCH_THREADS))

        processor = OwlViTProcessor.from_pretrained(MODEL_ID)
        model = OwlViTForObjectDetection.from_pretrained(MODEL_ID)
        device = self.device
        try:
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001
            if str(device).startswith("cuda"):
                logger.warning("GPU load failed (%s). Falling back to CPU.", exc)
                device = "cpu"
                model = model.to(device)
            else:
                raise
        self.device = device

        self.client.connect(MQTT_HOST, MQTT_PORT, 60)
        self.client.loop_start()
        try:
            while not stop_event.is_set():
                now = time.time()
                if not self.trigger.should_run(now):
                    time.sleep(0.05)
                    continue
                if not self.last_frame_bytes:
                    time.sleep(0.1)
                    continue
                if (now - self.last_frame_ts) > FRAME_TIMEOUT:
                    time.sleep(0.1)
                    continue
                img = self._decode_frame(self.last_frame_bytes)
                if img is None:
                    time.sleep(0.1)
                    continue
                img = self._resize(img)
                prompts = [entry[0] for entry in self.prompt_entries]
                categories = [entry[1] for entry in self.prompt_entries]
                inputs = processor(text=[prompts], images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                target_sizes = torch.tensor([img.size[::-1]], device=device)
                post_threshold = get_post_threshold()
                results = processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=post_threshold,
                )[0]

                anchors = []
                w, h = img.size
                for score_val, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
                    score = float(score_val.item())
                    if score < CONF_THRESHOLD:
                        continue
                    idx = int(label_idx.item())
                    category = categories[idx] if idx < len(categories) else "goal/salient"
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    x1n = max(0.0, min(1.0, x1 / w))
                    y1n = max(0.0, min(1.0, y1 / h))
                    x2n = max(0.0, min(1.0, x2 / w))
                    y2n = max(0.0, min(1.0, y2 / h))
                    anchors.append(
                        {
                            "label": category,
                            "score": round(score, 4),
                            "bbox": [round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)],
                            "center": [round((x1n + x2n) / 2.0, 4), round((y1n + y2n) / 2.0, 4)],
                        }
                    )

                limited: List[Dict[str, object]] = []
                per_label: Dict[str, int] = {}
                for anchor in sorted(anchors, key=lambda a: a.get("score", 0.0), reverse=True):
                    lbl = str(anchor.get("label") or "")
                    per_label[lbl] = per_label.get(lbl, 0) + 1
                    if per_label[lbl] <= MAX_PER_LABEL:
                        limited.append(anchor)

                self._publish(limited, self.last_frame_ts)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def handle_signal(_signum, _frame):
    stop_event.set()


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    AnchorDetector().run()


if __name__ == "__main__":
    main()
