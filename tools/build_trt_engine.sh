#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-}"
PRECISION="${2:-fp16}"

if [[ -z "$WEIGHTS" ]]; then
  echo "Usage: $0 <onnx_path> [fp16|int8]" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[error] ONNX not found: $WEIGHTS" >&2
  exit 1
fi

TRT_EXEC="${TRT_EXEC:-trtexec}"
IMG_W="${IMG_W:-640}"
IMG_H="${IMG_H:-640}"
MAX_BATCH="${MAX_BATCH:-1}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$WEIGHTS")}"
ENGINE_NAME="${ENGINE_NAME:-$(basename "${WEIGHTS%.*}")_${IMG_W}_${PRECISION}.engine}"
CALIB_DATA="${CALIB_DATA:-}"
CALIB_CACHE="${CALIB_CACHE:-${OUTPUT_DIR}/calibration.cache}"

mkdir -p "$OUTPUT_DIR"
ENGINE_PATH="${OUTPUT_DIR}/${ENGINE_NAME}"

COMMON_ARGS=(
  "--onnx=${WEIGHTS}"
  "--saveEngine=${ENGINE_PATH}"
  "--shapes=images:${MAX_BATCH}x3x${IMG_H}x${IMG_W}"
  "--workspace=${WORKSPACE_MB}"
  "--verbose"
)

if [[ "$PRECISION" == "fp16" ]]; then
  echo "[info] building FP16 engine: $ENGINE_PATH"
  "$TRT_EXEC" "${COMMON_ARGS[@]}" --fp16
elif [[ "$PRECISION" == "int8" ]]; then
  if [[ -z "$CALIB_DATA" ]]; then
    echo "[error] CALIB_DATA is required for int8 (directory of calibration images)" >&2
    exit 1
  fi
  echo "[info] building INT8 engine: $ENGINE_PATH"
  "$TRT_EXEC" "${COMMON_ARGS[@]}" --int8 --calib="${CALIB_DATA}" --calibCache="${CALIB_CACHE}"
else
  echo "[error] Unknown precision: $PRECISION (use fp16 or int8)" >&2
  exit 1
fi

echo "[ok] engine ready: ${ENGINE_PATH}"
