#!/usr/bin/env bash
set -euo pipefail

section() {
  echo
  echo "=== $1 ==="
}

section "nvmap audit"
echo "timestamp: $(date -Is)"
echo "host: $(hostname)"
echo "user: $(whoami)"

section "os / kernel"
if [[ -f /etc/nv_tegra_release ]]; then
  cat /etc/nv_tegra_release
else
  echo "/etc/nv_tegra_release: missing"
fi
uname -a
uname -r

section "disk"
df -h / || true

section "memory"
free -h || true
echo
echo "swap:"
swapon --show || true

section "jetpack / nvidia packages"
dpkg -l | grep -E "^ii\s+nvidia-(jetpack|l4t|cuda|tensorrt|cudnn|vpi|container)" || true

section "kernel modules"
lsmod | grep -E "(nvgpu|nvmap|nvidia)" || true

section "gpu stats (tegrastats, single sample)"
if command -v tegrastats >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    timeout 2 tegrastats --interval 1000 || true
  else
    echo "tegrastats: timeout not available; run manually: tegrastats --interval 1000"
  fi
else
  echo "tegrastats: not found"
fi

section "nvpmodel"
if command -v nvpmodel >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo nvpmodel -q || true
  else
    echo "nvpmodel: sudo required"
  fi
else
  echo "nvpmodel: not found"
fi

section "recent nvmap / nvgpu dmesg"
if sudo -n true 2>/dev/null; then
  sudo dmesg | tail -n 200 | grep -iE "(nvmap|nvgpu|nvhost|nvidia)" || true
elif dmesg >/dev/null 2>&1; then
  dmesg | tail -n 200 | grep -iE "(nvmap|nvgpu|nvhost|nvidia)" || true
else
  echo "dmesg: permission denied (run with sudo)"
fi

section "notes"
echo "- If release is R36.4.7, see docs/jetson/nvmapmemalloc_audit_playbook.md for NVIDIA's official workaround/patch steps."
echo "- This script is read-only and does not change system state."
