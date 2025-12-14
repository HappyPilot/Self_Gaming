#!/bin/bash
set -euo pipefail

if ! python3 -c "import pyds" >/dev/null 2>&1; then
  echo "[entrypoint] FATAL: pyds not available in image. Rebuild image to include pyds."
  exit 1
fi
exec python3 /app/deepstream_mqtt.py
