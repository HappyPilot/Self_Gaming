#!/bin/bash
set -euo pipefail

# Optional cache to repair pyds if the site-packages gets wiped
PYDS_CACHE="${PYDS_CACHE:-/mnt/ssd/models/pyds/pyds.so}"

if ! python3 -c "import pyds" >/dev/null 2>&1; then
  site_dir="$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

  if [ -f "$PYDS_CACHE" ]; then
    echo "[entrypoint] restoring pyds from cache $PYDS_CACHE"
    mkdir -p "$site_dir"
    cp "$PYDS_CACHE" "$site_dir"/pyds.so
  fi

  if ! python3 -c "import pyds" >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: pyds not present in image (and cache restore failed). Rebuild perception_ds image." >&2
    exit 1
  fi
fi

exec python3 /app/deepstream_mqtt.py
