#!/usr/bin/env python3
"""
Zero-shot detector using OWL-ViT.
Subscribes to vision/frame/preview, runs text-prompt detection, publishes to vision/objects.
"""
import json
import os
import queue
import signal
import threading
import time
from typing import List
import io

import numpy as np
import paho.mqtt.client as mqtt
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from utils.frame_transport import get_frame_bytes

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC", "vision/objects")
MODEL_ID = os.getenv("ZSD_MODEL_ID", "google/owlvit-base-patch32")
PROMPTS = [p.strip() for p in os.getenv("ZSD_PROMPTS", "enemy,boss,player,npc,loot,portal,waypoint,minimap,inventory,quest_marker,dialog_button,health_orb,mana_orb,menu_button").split(",") if p.strip()]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = float(os.getenv("ZSD_CONF_THRESHOLD", "0.25"))
MAX_PER_PROMPT = int(os.getenv("ZSD_MAX_PER_PROMPT", "2"))
INFER_INTERVAL = float(os.getenv("ZSD_INTERVAL", "2.0"))
FRAME_TIMEOUT = float(os.getenv("ZSD_FRAME_TIMEOUT", "5.0"))
MAX_SIZE = int(os.getenv("ZSD_MAX_SIZE", "640"))
TORCH_THREADS = int(os.getenv("ZSD_THREADS", "2"))

torch.set_num_threads(max(1, TORCH_THREADS))

stop_event = threading.Event()
frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=2)


def decode_frame(raw: bytes):
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception:
        return None


def on_connect(client, userdata, flags, rc):
    client.subscribe(FRAME_TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        raw = get_frame_bytes(payload)
        if not raw:
            return
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(raw)
    except Exception:
        pass


def publish_objects(client: mqtt.Client, objects: List[dict]):
    payload = {
        "ok": True,
        "timestamp": time.time(),
        "frame_ts": time.time(),
        "objects": objects,
        "backend": "zeroshot_owlvit",
    }
    client.publish(OBJECT_TOPIC, json.dumps(payload), qos=0)


def run():
    client = mqtt.Client(client_id="zeroshot_detector", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

    last_infer = 0.0
    last_frame_ts = time.time()

    while not stop_event.is_set():
        try:
            raw = frame_queue.get(timeout=0.5)
            last_frame_ts = time.time()
            if time.time() - last_infer < INFER_INTERVAL:
                continue

            img = decode_frame(raw)
            if img is None:
                continue

            if MAX_SIZE > 0:
                w, h = img.size
                longer = max(w, h)
                if longer > MAX_SIZE:
                    scale = MAX_SIZE / float(longer)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.BILINEAR)

            inputs = processor(text=[PROMPTS], images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]], device=DEVICE)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]

            if len(results["scores"]) > 0:
                max_score = float(results["scores"].max().item())
            else:
                max_score = 0.0
            print(f"[zsd] max_score={max_score:.4f} device={DEVICE}")

            objects = []
            for scores, labels, boxes in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(scores.item())
                if score < CONF_THRESHOLD:
                    continue
                label = PROMPTS[int(labels.item())]
                x1, y1, x2, y2 = boxes.tolist()
                w, h = img.size
                objects.append({
                    "class": label,
                    "confidence": round(score, 4),
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "box_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                    "backend": "zeroshot_owlvit",
                })
            limited = []
            per_label = {}
            for obj in sorted(objects, key=lambda o: o["confidence"], reverse=True):
                lbl = obj["class"]
                per_label[lbl] = per_label.get(lbl, 0) + 1
                if per_label[lbl] <= MAX_PER_PROMPT:
                    limited.append(obj)

            publish_objects(client, limited)
            last_infer = time.time()
        except queue.Empty:
            if time.time() - last_frame_ts > FRAME_TIMEOUT:
                time.sleep(0.1)
            continue
        except Exception:
            continue

    client.loop_stop()
    client.disconnect()


def handle_signal(signum, frame):
    stop_event.set()


if __name__ == "__main__":
    import cv2  # late import to avoid overhead if unused
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    run()
