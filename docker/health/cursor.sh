#!/usr/bin/env bash
set -euo pipefail

STATUS_FILE="${CURSOR_HEALTH_FILE:-/tmp/cursor_tracker_status.json}"
MAX_AGE_SEC="${CURSOR_HEALTH_MAX_AGE_SEC:-20}"

if [ ! -f "$STATUS_FILE" ]; then
  exit 1
fi

python3 - <<'PY'
import json
import os
import time

path = os.environ.get("CURSOR_HEALTH_FILE", "/tmp/cursor_tracker_status.json")
max_age = float(os.environ.get("CURSOR_HEALTH_MAX_AGE_SEC", "20"))

try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    raise SystemExit(1)

health = str(data.get("health") or "").lower()
last_ts = data.get("timestamp")
try:
    last_ts = float(last_ts)
except Exception:
    last_ts = None

now = time.time()
if last_ts is None or now - last_ts > max_age:
    raise SystemExit(1)

if health == "fail":
    raise SystemExit(1)

raise SystemExit(0)
PY
