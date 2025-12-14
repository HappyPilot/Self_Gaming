#!/usr/bin/env bash
set -euo pipefail
JETSON_HOST="${JETSON_HOST:-dima@10.0.0.68}"
JETSON_LOG_DIR="${JETSON_LOG_DIR:-/mnt/ssd/logs}"
DEST_DIR="${JETSON_LOGS_DEST:-$(cd "$(dirname "$0")/.." && pwd)/logs_jetson}"
mkdir -p "$DEST_DIR"
rsync -avz --delete "$JETSON_HOST":"$JETSON_LOG_DIR"/ "$DEST_DIR"/
echo "Synced $JETSON_HOST:$JETSON_LOG_DIR -> $DEST_DIR"
