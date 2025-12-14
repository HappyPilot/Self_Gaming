#!/bin/bash
set -euo pipefail

DEV="${DS_V4L2_DEVICE:-/dev/video0}"

echo "=== v4l2 devices ==="
v4l2-ctl --list-devices || true
echo

echo "=== v4l2-ctl --all (${DEV}) ==="
v4l2-ctl -d "${DEV}" --all | head -n 80 || true
echo

echo "=== v4l2-ctl --list-formats-ext (${DEV}) ==="
v4l2-ctl -d "${DEV}" --list-formats-ext | head -n 200 || true
echo

echo "=== open handles for /dev/video* (fuser) ==="
sudo sh -lc 'for v in /dev/video*; do echo "-- $v --"; fuser -v "$v" 2>/dev/null || true; done'
echo

echo "=== open handles (proc fd grep) ==="
sudo sh -lc 'grep -R "/dev/video" /proc/*/fd/* 2>/dev/null | head -n 50 || true'
echo

echo "=== possible camera consumers ==="
ps aux | egrep "pipewire|wireplumber|portal|gstreamer|gst-|ffmpeg|vlc|opencv|python" || true
