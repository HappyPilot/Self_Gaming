#!/usr/bin/env python3
"""Summarize recorder episodes on disk."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
PATTERN = os.getenv("RECORDER_PATTERN", "sample_*.json")


@dataclass
class EpisodeSummary:
    count: int
    total_bytes: int
    last_file: Optional[str]
    last_timestamp: Optional[float]
    last_stage: Optional[int]
    last_delta: Optional[str]

    def pretty(self) -> str:
        size_mb = self.total_bytes / (1024 * 1024)
        last_time = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_timestamp))
            if self.last_timestamp
            else "n/a"
        )
        return (
            f"Episode files: {self.count}\n"
            f"Total size: {size_mb:.2f} MB\n"
            f"Last sample: {self.last_file or 'n/a'}"
            f" (time={last_time}, stage={self.last_stage}, delta={self.last_delta})"
        )


def scan_episode_dir(directory: Path, pattern: str = PATTERN) -> EpisodeSummary:
    count = 0
    total = 0
    last_mtime = 0.0
    last_path: Optional[Path] = None
    for path in directory.glob(pattern):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total += stat.st_size
        if stat.st_mtime >= last_mtime:
            last_mtime = stat.st_mtime
            last_path = path
    last_file = None
    last_timestamp = None
    last_stage = None
    last_delta = None
    if last_path is not None:
        last_file = last_path.name
        try:
            payload: Dict[str, object] = json.loads(last_path.read_text(encoding="utf-8"))
            last_timestamp = float(payload.get("timestamp")) if payload.get("timestamp") else None
            last_stage = payload.get("stage")
            last_delta = payload.get("delta")
        except Exception:
            pass
    return EpisodeSummary(
        count=count,
        total_bytes=total,
        last_file=last_file,
        last_timestamp=last_timestamp,
        last_stage=last_stage,
        last_delta=last_delta,
    )


def main():
    parser = argparse.ArgumentParser(description="Show recorder dataset stats")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Episode directory to scan")
    parser.add_argument("--pattern", default=PATTERN, help="Filename pattern (default sample_*.json)")
    args = parser.parse_args()
    summary = scan_episode_dir(args.dir, args.pattern)
    print(summary.pretty())


if __name__ == "__main__":
    main()
