#!/usr/bin/env python3
"""Universal OCR fast path: run Tesseract on dynamic ROIs from ui_region_agent."""
from __future__ import annotations

import io
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List

import paho.mqtt.client as mqtt
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from utils.frame_transport import get_frame_bytes

# Logging / setup
logging.basicConfig(level=os.getenv("SIMPLE_OCR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("simple_ocr")
stop_event = threading.Event()

# Env
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv(
    "FRAME_TOPIC",
    os.getenv(
        "SIMPLE_OCR_FRAME_TOPIC",
        os.getenv("OCR_FRAME_TOPIC", os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")),
    ),
)
REGIONS_TOPIC = os.getenv("OCR_REGIONS_TOPIC", "ocr/regions")
OUTPUT_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")  # backward compatibility
UNIFIED_TOPIC = os.getenv("OCR_UNIFIED_TOPIC", "ocr/unified")
INTERVAL = float(os.getenv("SIMPLE_OCR_INTERVAL", "1.0"))
REGION_TTL_SEC = float(os.getenv("REGION_TTL_SEC", "5.0"))
PUBLISH_EMPTY = os.getenv("PUBLISH_EMPTY", "1") == "1"
MAX_REGIONS = int(os.getenv("MAX_REGIONS", "6"))
DEBUG_SAVE_ROI = os.getenv("DEBUG_SAVE_ROI", "0") == "1"
ROI_DEBUG_DIR = Path(os.getenv("ROI_DEBUG_DIR", "/tmp/ocr_roi_debug"))

# Preprocess
UPSCALE = float(os.getenv("SIMPLE_OCR_UPSCALE", "2.0"))
SHARPEN = float(os.getenv("SIMPLE_OCR_SHARPEN", "1.6"))
CONTRAST = float(os.getenv("SIMPLE_OCR_CONTRAST", "1.4"))
TESS_CONFIG = os.getenv("SIMPLE_OCR_TESS_CONFIG", "-l eng --psm 6")


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class SimpleOcrAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="simple_ocr", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.regions: List[Dict] = []
        self.regions_ts = 0.0
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            logger.warning("Failed to decode message on %s", msg.topic)
            return

        if msg.topic == FRAME_TOPIC:
            raw = get_frame_bytes(payload)
            if not raw:
                return
            with self.frame_lock:
                self.last_frame = raw
        elif msg.topic == REGIONS_TOPIC:
            regions = payload.get("regions") or []
            with self.frame_lock:
                self.regions = regions[:MAX_REGIONS]
                self.regions_ts = time.time()

    def _preprocess_crop(self, img: Image.Image, box: List[float], idx: int) -> Image.Image:
        w, h = img.size
        x, y, bw, bh = box
        x1, y1 = int(clamp01(x) * w), int(clamp01(y) * h)
        x2, y2 = int(clamp01(x + bw) * w), int(clamp01(y + bh) * h)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        crop = img.crop((x1, y1, x2, y2)).convert("L")
        if UPSCALE > 1.0:
            crop = crop.resize((int(crop.width * UPSCALE), int(crop.height * UPSCALE)), Image.BICUBIC)
        crop = ImageOps.autocontrast(crop, cutoff=2)
        crop = ImageEnhance.Contrast(crop).enhance(CONTRAST)
        crop = ImageEnhance.Sharpness(crop).enhance(SHARPEN)
        if DEBUG_SAVE_ROI:
            ROI_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            crop.save(ROI_DEBUG_DIR / f"roi_{ts}_{idx}.png")
        return crop

    def _run_ocr(self, raw: bytes, regions: List[Dict]) -> List[Dict]:
        img = Image.open(io.BytesIO(raw))
        items = []
        for idx, r in enumerate(regions[:MAX_REGIONS]):
            box = r.get("box")
            if not box or len(box) != 4:
                continue
            crop = self._preprocess_crop(img, [float(v) for v in box], idx)
            text = pytesseract.image_to_string(crop, config=TESS_CONFIG)
            cleaned = " ".join(text.split())
            items.append(
                {
                    "region_id": r.get("id", f"r{idx}"),
                    "text": cleaned,
                    "conf": None,
                    "box": box,
                    "source": "tesseract_roi",
                }
            )
        return items

    def _worker(self):
        while not stop_event.is_set():
            stop_event.wait(max(0.2, INTERVAL))
            if stop_event.is_set():
                break
            with self.frame_lock:
                raw = self.last_frame
                regions = self.regions
                rts = self.regions_ts
            now = time.time()
            fresh_regions = regions if (now - rts) <= REGION_TTL_SEC else []
            if raw is None:
                continue
            try:
                items = self._run_ocr(raw, fresh_regions)
                items = [it for it in items if it.get("text")]
                if not items and not PUBLISH_EMPTY:
                    continue
                payload = {"ts": now, "items": items, "heartbeat": True}
                self.client.publish(UNIFIED_TOPIC, json.dumps(payload))
                # Backward compatibility
                self.client.publish(OUTPUT_TOPIC, json.dumps(payload))
                if items:
                    logger.info("Published %d ROI texts", len(items))
                elif not PUBLISH_EMPTY:
                    # If heartbeat disabled, skip empty publish
                    continue
            except Exception as exc:
                logger.error("OCR worker error: %s", exc)

    def _on_connect(self, cli, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            cli.subscribe([(FRAME_TOPIC, 0), (REGIONS_TOPIC, 0)])
            logger.info("Connected and subscribed to %s and %s", FRAME_TOPIC, REGIONS_TOPIC)
        else:
            logger.error("Connect failed rc=%s", _as_int(rc))

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.worker_thread.start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected")


def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down.", signum)
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = SimpleOcrAgent()
    agent.start()


if __name__ == "__main__":
    main()
