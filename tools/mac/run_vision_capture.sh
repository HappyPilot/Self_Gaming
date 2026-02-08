#!/usr/bin/env bash
set -euo pipefail

# Publish screen frames from macOS to Jetson MQTT.
# Requires: pip3 install --user mss

JETSON_IP=${JETSON_IP:-10.0.0.68}
MQTT_PORT=${MQTT_PORT:-1883}
MSS_MONITOR=${MSS_MONITOR:-1}

export MQTT_HOST="$JETSON_IP"
export MQTT_PORT="$MQTT_PORT"
export CAPTURE_BACKEND="screen"
export CAPTURE_BACKEND_FALLBACKS="screen,mss"
export MSS_MONITOR="$MSS_MONITOR"

export VISION_FRAME_TOPIC=${VISION_FRAME_TOPIC:-vision/frame/preview}
export VISION_FRAME_PREVIEW_TOPIC=${VISION_FRAME_PREVIEW_TOPIC:-vision/frame/preview}
export VISION_FRAME_FULL_TOPIC=${VISION_FRAME_FULL_TOPIC:-vision/frame/full}
export VISION_FRAME_INTERVAL=${VISION_FRAME_INTERVAL:-0.2}
export VISION_FRAME_JPEG_QUALITY=${VISION_FRAME_JPEG_QUALITY:-60}
export VISION_FRAME_PREVIEW_QUALITY=${VISION_FRAME_PREVIEW_QUALITY:-60}
export VISION_FRAME_FULL_QUALITY=${VISION_FRAME_FULL_QUALITY:-85}

python3 /Users/dima/self-gaming/agents/vision_agent.py
