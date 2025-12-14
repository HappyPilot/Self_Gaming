#!/usr/bin/env python3
"""Lightweight YOLO-World debug runner for container sanity checks.

Usage inside the perception container:
  python3 /app/tools/debug_yoloworld.py --weights /mnt/ssd/models/yolo/yolov8s-world.pt --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=os.getenv("YOLO11_WEIGHTS", "/mnt/ssd/models/yolo/yolov8s-world.pt"))
    parser.add_argument("--classes", default=os.getenv("YOLO_WORLD_CLASSES", "enemy,boss,player,npc,loot,portal,waypoint"))
    parser.add_argument("--device", default=os.getenv("YOLO11_DEVICE", "cuda:0"))
    parser.add_argument("--conf", type=float, default=float(os.getenv("YOLO11_CONF", "0.1")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("YOLO11_IMGSZ", "640")))
    parser.add_argument("--max-size", type=int, default=int(os.getenv("YOLO_MAX_SIZE", "0")), help="Optional pre-resize long side")
    parser.add_argument("--image", type=Path, help="Optional image path for inference; defaults to a blank frame")
    parser.add_argument("--clip-cpu", action="store_true", default=True, help="Run CLIP text encoder on CPU to avoid CUDA allocator issues")
    return parser.parse_args()


@contextmanager
def clip_on_cpu():
    orig = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False  # type: ignore
        yield
    finally:
        torch.cuda.is_available = orig


def load_frame(image_path: Path | None, max_size: int) -> np.ndarray:
    if image_path and image_path.exists():
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image at {image_path}")
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    if max_size and max(frame.shape[:2]) > max_size:
        h, w = frame.shape[:2]
        scale = float(max_size) / float(max(h, w))
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def main() -> int:
    args = parse_args()
    classes = [cls.strip() for cls in args.classes.split(",") if cls.strip()]
    if args.device.lower() == "cpu":
        # Force CPU path even if CUDA is available to avoid allocator crashes.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False  # type: ignore
    print(f"torch.cuda.is_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    print(f"Loading YOLO-World weights={args.weights} device={args.device} classes={classes}")
    model = YOLOWorld(args.weights)
    if args.clip_cpu:
        with clip_on_cpu():
            model.set_classes(classes)
    else:
        model.set_classes(classes)

    frame = load_frame(args.image, args.max_size)
    print(f"Frame shape={frame.shape} dtype={frame.dtype}")
    try:
        results = model.predict(source=frame, device=args.device, conf=args.conf, imgsz=args.imgsz, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] inference failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No results returned")
        return 0

    res = results[0]
    boxes = getattr(res, "boxes", None)
    num = 0 if boxes is None or boxes.xyxy is None else len(boxes.xyxy)
    max_score = 0.0 if boxes is None or boxes.conf is None or len(boxes.conf) == 0 else float(boxes.conf.max())
    print(f"Detections: {num}, max_score={max_score:.4f}, names={res.names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
