#!/usr/bin/env python3
"""External heavy-vision server for open-vocab hints.

Run on the Mac/PC with GPU:
    python3 tools/hint_server.py --port 5010 --weights /mnt/ssd/models/yolo/yolov8s-world.pt
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLOWorld

logging.basicConfig(level=os.getenv("HINT_SERVER_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("hint_server")


def decode_image(image_b64: str) -> np.ndarray | None:
    try:
        buf = base64.b64decode(image_b64)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        logger.exception("Failed to decode image")
        return None


def detections_to_dicts(result, frame_shape: Tuple[int, int]) -> List[dict]:
    h, w = frame_shape[:2]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    names = result.names or {}
    out = []
    raw_boxes = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    for (x1, y1, x2, y2), conf, cls in zip(raw_boxes, scores, classes):
        out.append(
            {
                "label": names.get(int(cls), f"cls_{int(cls)}") if isinstance(names, dict) else str(int(cls)),
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_norm": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
            }
        )
    return out


def build_model(weights: str, classes: list[str], device: str):
    model = YOLOWorld(weights)
    model.set_classes(classes)
    return model, device


def create_app(model, device: str, conf: float, imgsz: int):
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True})

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        frame_id = payload.get("frame_id")
        if not image_b64:
            return jsonify({"ok": False, "error": "no image"}), 400
        frame = decode_image(image_b64)
        if frame is None:
            return jsonify({"ok": False, "error": "decode failed"}), 400
        results = model.predict(source=frame, device=device, conf=conf, imgsz=imgsz, verbose=False)
        detections = detections_to_dicts(results[0], frame.shape) if results else []
        return jsonify({"ok": True, "frame_id": frame_id, "detections": detections})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--weights", default=os.getenv("HINT_WEIGHTS", "/mnt/ssd/models/yolo/yolov8s-world.pt"))
    parser.add_argument("--classes", default=os.getenv("HINT_CLASSES", "enemy,boss,player,npc,loot,portal,waypoint,quest_marker,dialog_button"))
    parser.add_argument("--device", default=os.getenv("HINT_DEVICE", "cuda:0"))
    parser.add_argument("--conf", type=float, default=float(os.getenv("HINT_CONF", "0.1")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("HINT_IMGSZ", "640")))
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    logger.info("Loading model weights=%s device=%s classes=%s", args.weights, args.device, classes)
    model, device = build_model(args.weights, classes, args.device)
    app = create_app(model, device, conf=args.conf, imgsz=args.imgsz)
    logger.info("Hint server listening on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
