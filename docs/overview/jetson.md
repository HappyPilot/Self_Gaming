# Jetson Environment Notes

Captured from the main Jetson host on 2026-01-31.

## System
- Hostname: ubuntu
- Kernel: Linux 5.15.148-tegra (aarch64)
- Python: 3.10.12

## Repo
- Path: /home/dima/self-gaming
- Branch: pr-nitrogen-skeleton
- Latest commit: a600457 ("Use UI layout for target filtering and fallback skill keys")
- Untracked: agents/data/prompt_profiles.json

## Docker
- Docker: 29.0.0 (build 3d4129b)
- Docker Compose: v2.40.3

## Notes
- python3 -m unittest ran zero tests and exited OK on Jetson.
- OCR paddle agent emitted empty/unchanged_frame logs during the run (expected if frame unchanged).
