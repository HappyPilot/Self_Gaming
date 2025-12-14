#!/bin/bash
set -euo pipefail

DEV="${DS_V4L2_DEVICE:-/dev/video0}"

echo "[reset] target device: ${DEV}"

echo "[reset] stopping common desktop camera services (best-effort)..."
systemctl --user stop pipewire pipewire-pulse wireplumber 2>/dev/null || true
systemctl stop pipewire pipewire-pulse wireplumber 2>/dev/null || true

usb_path=""
if command -v udevadm >/dev/null 2>&1; then
  usb_path=$(udevadm info -q path -n "${DEV}" 2>/dev/null || true)
fi

if [[ -n "${usb_path}" ]]; then
  usb_id=$(echo "${usb_path}" | sed 's#.*/usb/##' | cut -d/ -f1)
  if [[ -n "${usb_id}" && -e "/sys/bus/usb/drivers/usb/${usb_id}" ]]; then
    echo "[reset] unbind/bind USB device ${usb_id}"
    echo "${usb_id}" | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null || true
    sleep 1
    echo "${usb_id}" | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null || true
  else
    echo "[reset] usb id not found for ${DEV}, skipping unbind/bind"
  fi
else
  echo "[reset] udevadm path not found for ${DEV}, skipping unbind/bind"
fi

echo "[reset] trying uvcvideo reload (may fail if busy)..."
sudo modprobe -r uvcvideo 2>/dev/null || true
sudo modprobe uvcvideo 2>/dev/null || true

echo "[reset] quick stream test..."
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl -d "${DEV}" --stream-mmap=3 --stream-count=30 --stream-to=/dev/null || true
else
  echo "[reset] v4l2-ctl not available in PATH"
fi

echo "[reset] If still EBUSY: physically replug capture card or reboot."
