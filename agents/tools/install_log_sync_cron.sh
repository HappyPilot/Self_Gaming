#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SYNC_SCRIPT="$SCRIPT_DIR/sync_jetson_logs.sh"
LOG_FILE="$SCRIPT_DIR/../log_sync_cron.log"
if [[ ! -x "$SYNC_SCRIPT" ]]; then
  echo "Sync script $SYNC_SCRIPT is not executable" >&2
  exit 1
fi
CRON_LINE="0 * * * * $SYNC_SCRIPT >> $LOG_FILE 2>&1"
TMP=$(mktemp)
crontab -l 2>/dev/null | grep -v "$SYNC_SCRIPT" > "$TMP" || true
echo "$CRON_LINE" >> "$TMP"
crontab "$TMP"
rm "$TMP"
echo "Installed hourly cron job. Logs -> $LOG_FILE"
