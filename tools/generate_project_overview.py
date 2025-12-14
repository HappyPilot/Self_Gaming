#!/usr/bin/env python3
"""Generate PROJECT_OVERVIEW.md with services, topics, and config snapshot."""
from __future__ import annotations

import datetime
import pathlib
import re
import subprocess
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "PROJECT_OVERVIEW.md"


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, shell=True, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:  # pragma: no cover
        return f"[command failed] {cmd}\n{exc}\n"


def scan_topics() -> list[str]:
    topics = set()
    for p in ROOT.rglob("*.py"):
        try:
            s = p.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.findall(r"['\"]([A-Za-z0-9_-]+/[A-Za-z0-9_/#-]+)['\"]", s):
            if "/" in m and len(m) <= 120:
                topics.add(m)
    return sorted(topics)


def main():
    now = datetime.datetime.now().isoformat(timespec="seconds")
    top_dirs = sorted([x.name for x in ROOT.iterdir() if x.is_dir() and not x.name.startswith(".")])

    git_rev = sh("git rev-parse --short HEAD || true").strip()
    compose_ps = sh("docker compose ps || true").strip()
    compose_cfg = sh("docker compose config || true")

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "compose_config.yaml").write_text(compose_cfg)

    topics = scan_topics()

    OUT.write_text(
        textwrap.dedent(
            f"""\
            # Project Overview (auto-generated)
            Generated: {now}
            Git: {git_rev}

            ## What this repo is
            Self-Gaming multi-agent stack (vision/OCR/policy/act) with Jetson + DeepStream + MQTT.

            ## Top-level directories
            {chr(10).join(f"- `{d}/`" for d in top_dirs)}

            ## Docker Compose services (current)
            ```text
            {compose_ps}
            ```

            ## Compose config snapshot
            Saved to: `docs/compose_config.yaml`

            ## MQTT topics discovered (best-effort scan)
            {chr(10).join(f"- `{t}`" for t in topics)}

            ## Data flow (expected)
            Video -> Vision/DeepStream -> OCR (radar+ROI) -> Scene aggregation -> Policy/Teacher -> Act bridge.

            ## Modes
            - `DS_SOURCE=v4l2` uses `/dev/videoX` capture; HTTP viewer optional.
            - OCR radar: low-frequency Paddle via ocr_easy; ROI OCR via simple_ocr; unified stream `ocr/unified`.
            - Policy brain: shadow mode publishes `policy_brain/cmd`, metrics on `policy_brain/metrics`.

            ## Troubleshooting
            - V4L2 device mismatch: check `/dev/video*`, set `DS_V4L2_DEVICE`, restart perception_ds.
            - DeepStream restarts: verify `DS_SOURCE` and caps, see `docker logs perception_ds`.
            - OCR high CPU: ensure thread caps env, radar interval, and frame-hash gating.
            - GPU idle: ensure pgie running and GR3D_FREQ > 0 in `tegrastats`.

            ## Health snapshot commands
            - `free -h; swapon --show`
            - `timeout 10s tegrastats`
            - `docker stats --no-stream`
            - `docker compose ps`
            """
        ).strip()
        + "\n"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
