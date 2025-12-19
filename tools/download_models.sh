#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/ssd/models/yolo}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found. Install it or run inside the perception container." >&2
  exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1; then
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ultralytics") else 1)
PY
  echo "[error] ultralytics is not installed. Run: pip install ultralytics" >&2
  echo "[hint] If you're on Jetson, run this inside the container where ultralytics is available." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

echo "[info] target dir: $MODEL_DIR"

echo "[info] downloading YOLO11 weights via ultralytics"
python3 tools/download_yolo11_weights.py yolo11n.pt --output "$MODEL_DIR"
python3 tools/download_yolo11_weights.py yolov8s-world.pt --output "$MODEL_DIR"

echo "[ok] models available in $MODEL_DIR"
