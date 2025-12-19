#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/ssd/models/yolo}"

mkdir -p "$MODEL_DIR"

echo "[info] target dir: $MODEL_DIR"

echo "[info] downloading YOLO11 weights via ultralytics"
python3 tools/download_yolo11_weights.py yolo11n.pt --output "$MODEL_DIR"
python3 tools/download_yolo11_weights.py yolov8s-world.pt --output "$MODEL_DIR"

echo "[ok] models available in $MODEL_DIR"
