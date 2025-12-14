#!/usr/bin/env python3
"""Generate heuristic YOLO labels for PoE frames."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["player", "enemy", "health_orb", "mana_orb"]


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label PoE screenshots")
    parser.add_argument("--images", required=True, help="Path to raw frames (directory)")
    parser.add_argument("--output", default="data/poe_yolo", help="Dataset root output")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(root: Path):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def save_label(path: Path, boxes: list[tuple[int, float, float, float, float]]):
    with open(path, "w", encoding="utf-8") as handle:
        for cls, xc, yc, w, h in boxes:
            handle.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def detect_enemies(img: np.ndarray) -> list[tuple[float, float, float, float]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 120, 80])
    upper2 = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = img.shape[:2]
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 500:  # filter tiny noise
            continue
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        boxes.append((xc, yc, bw / w, bh / h))
    return boxes


def heuristics_for_frame(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    boxes = []
    # player near center
    player_w = 0.18
    player_h = 0.3
    boxes.append((0, 0.5, 0.55, player_w, player_h))
    # health orb bottom-left
    boxes.append((2, 0.1, 0.85, 0.2, 0.25))
    # mana orb bottom-right
    boxes.append((3, 0.9, 0.85, 0.2, 0.25))
    for xc, yc, bw, bh in detect_enemies(img):
        boxes.append((1, xc, yc, bw, bh))
    return boxes


def main():
    args = parse_args()
    image_paths = sorted((Path(args.images)).glob("*.jpg"))
    if not image_paths:
        raise SystemExit("No images found")
    random.seed(args.seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * args.train_split)
    splits = {"train": image_paths[:split_idx], "val": image_paths[split_idx:]}
    root = Path(args.output)
    ensure_dirs(root)

    for split, files in splits.items():
        for img_path in files:
            rel_name = img_path.name
            boxes = heuristics_for_frame(img_path)
            target_img = root / "images" / split / rel_name
            target_lbl = root / "labels" / split / (img_path.stem + ".txt")
            target_img.write_bytes(img_path.read_bytes())
            save_label(target_lbl, boxes)

    names_txt = "\n".join(CLASSES)
    data_yaml = f"""path: {root}
train: images/train
val: images/val
names:
  0: player
  1: enemy
  2: health_orb
  3: mana_orb
"""
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Dataset ready under {root} with {len(image_paths)} images")


if __name__ == "__main__":
    main()
