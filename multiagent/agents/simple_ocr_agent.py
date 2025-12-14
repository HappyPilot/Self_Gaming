#!/usr/bin/env python3
"""Lightweight OCR backup using pytesseract on a cropped region."""
import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("SIMPLE_OCR_FRAME_TOPIC", "vision/frame")
OUTPUT_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
INTERVAL = float(os.getenv("SIMPLE_OCR_INTERVAL", "5"))
CROP_LEFT = float(os.getenv("SIMPLE_OCR_CROP_LEFT", "0.35"))
CROP_RIGHT = float(os.getenv("SIMPLE_OCR_CROP_RIGHT", "0.95"))
CROP_TOP = float(os.getenv("SIMPLE_OCR_CROP_TOP", "0.25"))
CROP_BOTTOM = float(os.getenv("SIMPLE_OCR_CROP_BOTTOM", "0.85"))
UPSCALE = float(os.getenv("SIMPLE_OCR_UPSCALE", "2.5"))
SHARPEN = float(os.getenv("SIMPLE_OCR_SHARPEN", "1.8"))
CONTRAST = float(os.getenv("SIMPLE_OCR_CONTRAST", "1.5"))
LOG_DIR = Path(os.getenv("SIMPLE_OCR_DEBUG_DIR", "/tmp/simple_ocr"))
DEBUG = os.getenv("SIMPLE_OCR_DEBUG", "0") == "1"

last_frame = None
frame_lock = threading.Lock()

def log(msg: str):
    if DEBUG:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[simple_ocr] {msg}", flush=True)


def on_frame(_client, _userdata, msg):
    global last_frame
    try:
        payload = msg.payload.decode("utf-8", "ignore")
        data = json.loads(payload)
        img_b64 = data.get("image_b64")
        if not img_b64:
            return
        raw = base64.b64decode(img_b64)
        with frame_lock:
            last_frame = raw
    except Exception as exc:  # pragma: no cover - defensive logging
        log(f"frame decode failed: {exc}")


def preprocess(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    left = int(max(0, min(w, CROP_LEFT * w)))
    right = int(max(left + 1, min(w, CROP_RIGHT * w)))
    top = int(max(0, min(h, CROP_TOP * h)))
    bottom = int(max(top + 1, min(h, CROP_BOTTOM * h)))
    crop = img.crop((left, top, right, bottom)).convert("L")
    if UPSCALE > 1.0:
        crop = crop.resize((int(crop.width * UPSCALE), int(crop.height * UPSCALE)), Image.BICUBIC)
    crop = ImageOps.autocontrast(crop, cutoff=2)
    crop = ImageEnhance.Contrast(crop).enhance(CONTRAST)
    crop = ImageEnhance.Sharpness(crop).enhance(SHARPEN)
    if DEBUG:
        ts = int(time.time() * 1000)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        crop.save(LOG_DIR / f"simple_ocr_{ts}.png")
    return crop


def worker(client: mqtt.Client):
    while True:
        time.sleep(max(0.5, INTERVAL))
        with frame_lock:
            raw = last_frame
        if not raw:
            continue
        try:
            crop = preprocess(raw)
            text = pytesseract.image_to_string(crop, lang="eng")
            cleaned = " ".join(text.split())
            if cleaned:
                payload = json.dumps({"ok": True, "text": cleaned})
                client.publish(OUTPUT_TOPIC, payload)
                log(f"published {len(cleaned)} chars")
        except Exception as exc:  # pragma: no cover - runtime guard
            log(f"ocr error: {exc}")


def main():
    client = mqtt.Client(client_id="simple_ocr", protocol=mqtt.MQTTv311)

    def on_connect(cli, _userdata, _flags, rc):
        if rc == 0:
            cli.subscribe(FRAME_TOPIC)
            log("connected and subscribed")
        else:
            log(f"connect failed rc={rc}")

    client.on_connect = on_connect
    client.on_message = on_frame
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    threading.Thread(target=worker, args=(client,), daemon=True).start()
    client.loop_forever()


if __name__ == "__main__":
    main()
