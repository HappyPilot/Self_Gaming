#!/usr/bin/env bash
set -euo pipefail

MLFLOW_DIR="${MLFLOW_DIR:-/mnt/ssd/mlflow}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_BACKEND_URI="${MLFLOW_BACKEND_URI:-${MLFLOW_DIR}}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${MLFLOW_DIR}/artifacts}"

if ! command -v mlflow >/dev/null 2>&1; then
  echo "mlflow not found. Install with: pip install mlflow" >&2
  exit 1
fi

if ! mkdir -p "${MLFLOW_DIR}" 2>/dev/null; then
  MLFLOW_DIR="${PWD}/mlruns"
  MLFLOW_BACKEND_URI="${MLFLOW_DIR}"
  MLFLOW_ARTIFACT_ROOT="${MLFLOW_DIR}/artifacts"
  mkdir -p "${MLFLOW_DIR}"
fi

echo "Starting MLflow server"
echo "  backend: ${MLFLOW_BACKEND_URI}"
echo "  artifacts: ${MLFLOW_ARTIFACT_ROOT}"
echo "  host: ${MLFLOW_HOST}"
echo "  port: ${MLFLOW_PORT}"

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
