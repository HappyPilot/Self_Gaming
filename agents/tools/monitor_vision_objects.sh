#!/usr/bin/env bash
set -euo pipefail
MQTT_HOST="${MQTT_HOST:-127.0.0.1}"
MQTT_PORT="${MQTT_PORT:-1883}"
mosquitto_sub -h "$MQTT_HOST" -p "$MQTT_PORT" -v -t vision/objects
