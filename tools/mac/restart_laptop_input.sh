#!/usr/bin/env bash
set -euo pipefail

AGENT_PATH="${INPUT_AGENT_PATH:-/Users/dima/self-gaming/tools/mac/laptop_input_agent.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${JETSON_IP:=10.0.0.68}"
: "${MQTT_PORT:=1883}"
: "${ACT_TOPIC:=act/cmd}"
: "${INPUT_CONTROL_TOPIC:=act/control}"
: "${INPUT_HTTP_ENABLED:=1}"
: "${INPUT_HTTP_PORT:=5010}"
: "${INPUT_REQUIRE_FRONT_APP:=1}"
: "${INPUT_FRONT_APP:=auto}"
: "${INPUT_BOUNDS:=auto}"
: "${INPUT_PUBLISH_IDENTITY:=1}"
: "${INPUT_IDENTITY_PUBLISH_SEC:=5.0}"
: "${INPUT_LOG_LEVEL:=INFO}"
: "${INPUT_LOG_FILE:=/Users/dima/logs/laptop_input_agent.log}"
: "${INPUT_PID_FILE:=/tmp/laptop_input_agent.pid}"

LOG_DIR="$(dirname "${INPUT_LOG_FILE}")"
mkdir -p "${LOG_DIR}"

if pgrep -f "${AGENT_PATH}" >/dev/null 2>&1; then
  echo "Stopping existing input agent..."
  pkill -f "${AGENT_PATH}" || true
  sleep 0.5
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"${INPUT_HTTP_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${INPUT_HTTP_PORT} is still in use. Stop the process and retry."
    exit 1
  fi
fi

export JETSON_IP MQTT_PORT ACT_TOPIC INPUT_CONTROL_TOPIC INPUT_HTTP_ENABLED INPUT_HTTP_PORT
export INPUT_REQUIRE_FRONT_APP INPUT_FRONT_APP INPUT_BOUNDS
export INPUT_PUBLISH_IDENTITY INPUT_IDENTITY_PUBLISH_SEC
export INPUT_LOG_LEVEL INPUT_LOG_FILE

nohup "${PYTHON_BIN}" "${AGENT_PATH}" >> "${INPUT_LOG_FILE}" 2>&1 &
echo $! > "${INPUT_PID_FILE}" || true
echo "Started input bridge (pid=$(cat "${INPUT_PID_FILE}" 2>/dev/null || echo "unknown"))"
echo "Log: ${INPUT_LOG_FILE}"
