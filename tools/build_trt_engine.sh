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
INPUT_NAME="${INPUT_NAME:-images}"
IMG_W="${IMG_W:-640}"
IMG_H="${IMG_H:-640}"
MAX_BATCH="${MAX_BATCH:-1}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
WORKSPACE_BYTES=$((WORKSPACE_MB * 1024 * 1024))
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$WEIGHTS")}"
ENGINE_NAME="${ENGINE_NAME:-$(basename "${WEIGHTS%.*}")_${IMG_W}x${IMG_H}_${PRECISION}.engine}"
CALIB_DATA="${CALIB_DATA:-}"
CALIB_CACHE="${CALIB_CACHE:-${OUTPUT_DIR}/calibration.cache}"
FORCE_CALIB="${FORCE_CALIB:-0}"
EXPLICIT_BATCH="${EXPLICIT_BATCH:-1}"
FORCE="${FORCE:-0}"
CALIB_FLAG="${CALIB_FLAG:-calib}"

mkdir -p "$OUTPUT_DIR"
ENGINE_PATH="${OUTPUT_DIR}/${ENGINE_NAME}"

echo "[info] input=${INPUT_NAME} shape=${MAX_BATCH}x3x${IMG_H}x${IMG_W}"
echo "[info] workspace=${WORKSPACE_MB}MB (${WORKSPACE_BYTES} bytes)"
echo "[info] batch=${MAX_BATCH} (fixed)"

if [[ -f "$ENGINE_PATH" && "$FORCE" != "1" ]]; then
  echo "[skip] engine exists: $ENGINE_PATH (set FORCE=1 to rebuild)"
  exit 0
fi

COMMON_ARGS=(
  "--onnx=${WEIGHTS}"
  "--saveEngine=${ENGINE_PATH}"
  "--minShapes=${INPUT_NAME}:${MAX_BATCH}x3x${IMG_H}x${IMG_W}"
  "--optShapes=${INPUT_NAME}:${MAX_BATCH}x3x${IMG_H}x${IMG_W}"
  "--maxShapes=${INPUT_NAME}:${MAX_BATCH}x3x${IMG_H}x${IMG_W}"
  "--memPoolSize=workspace:${WORKSPACE_BYTES}"
  "--verbose"
)
if [[ "$EXPLICIT_BATCH" == "1" ]]; then
  COMMON_ARGS+=("--explicitBatch")
fi

if [[ "$PRECISION" == "fp16" ]]; then
  echo "[info] building FP16 engine: $ENGINE_PATH"
  "$TRT_EXEC" "${COMMON_ARGS[@]}" --fp16
elif [[ "$PRECISION" == "int8" ]]; then
  if [[ -n "$CALIB_DATA" ]]; then
    echo "[error] INT8 requires a calibration cache file. Provide CALIB_CACHE=/path/to/calibration.cache" >&2
    echo "[hint] Check TRT flags: $TRT_EXEC --help | grep -i calib" >&2
    exit 1
  fi
  if [[ ! -f "$CALIB_CACHE" && "$FORCE_CALIB" != "1" ]]; then
    echo "[error] Calibration cache not found: $CALIB_CACHE" >&2
    echo "[hint] Provide CALIB_CACHE=/path/to/calibration.cache" >&2
    echo "[hint] Check TRT flags: $TRT_EXEC --help | grep -i calib" >&2
    echo "[hint] Set FORCE_CALIB=1 to bypass this check" >&2
    exit 1
  fi
  if [[ "$FORCE_CALIB" == "1" ]]; then
    echo "[warn] FORCE_CALIB=1 set; skipping cache existence check" >&2
  fi
  echo "[info] building INT8 engine: $ENGINE_PATH"
  "$TRT_EXEC" "${COMMON_ARGS[@]}" --int8 "--${CALIB_FLAG}=${CALIB_CACHE}"
else
  echo "[error] Unknown precision: $PRECISION (use fp16 or int8)" >&2
  exit 1
fi

echo "[ok] engine ready: ${ENGINE_PATH}"
