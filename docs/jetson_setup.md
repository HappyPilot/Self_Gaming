# Jetson Setup

This guide covers the baseline steps to get the stack running on Jetson with sane performance defaults.

## Prereqs
- JetPack 6.x installed (Ubuntu 22.04 base).
- An SSD mounted at `/mnt/ssd` for models and logs.
- Docker + NVIDIA Container Runtime enabled.

Check JetPack version:
```bash
cat /etc/nv_tegra_release
```

## Performance Mode
List available power modes, then select MAXN for your SKU (requires root):
```bash
sudo nvpmodel -q
```

Set the MAXN mode by index:
```bash
sudo nvpmodel -m <maxn-index>
sudo jetson_clocks
```

Verify:
```bash
sudo jetson_clocks --show
```

## ZRAM
Ensure swap is available so short spikes do not OOM:
```bash
swapon --show
```

If ZRAM is disabled, enable the default config:
```bash
sudo systemctl enable --now nvzramconfig
```

If the unit name differs on your JetPack build:
```bash
systemctl status nvzramconfig
systemctl list-unit-files | grep -i zram
```

## Docker + NVIDIA Runtime
Confirm the NVIDIA runtime is available:
```bash
docker info | grep -n "Runtimes"
```

## Repo Profile
Use the Jetson profile:
```bash
export SG_PROFILE=jetson
```

Notes:
- `OCR_FORCE_CPU=1` overrides `OCR_USE_GPU=1`. Keep `OCR_FORCE_CPU=0` if you want GPU OCR.

If you need host-specific overrides, create `config/local.env` and export:
```bash
export SG_LOCAL_ENV_FILE="$(pwd)/config/local.env"
```

## SHM Transport (Optional)
If you enable `FRAME_TRANSPORT=shm`, every container that reads frames must share IPC:
- use `ipc: host` (or `ipc: "service:mq"`) for vision and all consumers
- set `shm_size: 1g` (or similar) in compose

If IPC is not shared, consumers will log "SHM segment not found" and drop frames.
