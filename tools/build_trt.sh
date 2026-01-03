#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-}"
ENGINE_PATH="${2:-}"

if [[ -z "$ONNX_PATH" || -z "$ENGINE_PATH" ]]; then
  echo "Usage: $0 /path/model.onnx /path/model.engine" >&2
  echo "Env: INPUT_NAME=input SHAPES=1x3x224x224 WORKSPACE_MB=2048 FP16=1" >&2
  exit 2
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "[error] ONNX not found: $ONNX_PATH" >&2
  exit 3
fi

TRT_EXEC="${TRT_EXEC:-trtexec}"
INPUT_NAME="${INPUT_NAME:-input}"
SHAPES="${SHAPES:-1x3x224x224}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
WORKSPACE_BYTES=$((WORKSPACE_MB * 1024 * 1024))
FP16="${FP16:-1}"
EXPLICIT_BATCH="${EXPLICIT_BATCH:-1}"

mkdir -p "$(dirname "$ENGINE_PATH")"

COMMON_ARGS=(
  "--onnx=${ONNX_PATH}"
  "--saveEngine=${ENGINE_PATH}"
  "--minShapes=${INPUT_NAME}:${SHAPES}"
  "--optShapes=${INPUT_NAME}:${SHAPES}"
  "--maxShapes=${INPUT_NAME}:${SHAPES}"
  "--memPoolSize=workspace:${WORKSPACE_BYTES}"
  "--verbose"
)

if [[ "$EXPLICIT_BATCH" == "1" ]]; then
  COMMON_ARGS+=("--explicitBatch")
fi

if [[ "$FP16" == "1" ]]; then
  COMMON_ARGS+=("--fp16")
fi

echo "[build_trt] ${TRT_EXEC} ${COMMON_ARGS[*]}"
"$TRT_EXEC" "${COMMON_ARGS[@]}"

if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "[error] Engine not created: $ENGINE_PATH" >&2
  exit 4
fi

echo "[ok] engine ready: $ENGINE_PATH"
