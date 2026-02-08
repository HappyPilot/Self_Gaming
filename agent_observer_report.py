#!/usr/bin/env python3
"""Summarize agent_observer snapshots and action windows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.utils import observer_report as report


def _load_samples(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize agent_observer snapshots")
    parser.add_argument("--dir", dest="out_dir", required=True, help="agent_observer output directory")
    parser.add_argument("--expect", type=int, default=6, help="expected sample count")
    parser.add_argument("--wait", action="store_true", help="wait until samples.json has expected count")
    parser.add_argument("--interval", type=float, default=10.0, help="poll interval seconds when waiting")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    samples_path = out_dir / "samples.json"

    if args.wait:
        while True:
            samples = _load_samples(samples_path)
            if len(samples) >= args.expect:
                break
            time.sleep(args.interval)

    samples = _load_samples(samples_path)
    reports = report.summarize_samples(samples)
    report.write_report(out_dir, reports)


if __name__ == "__main__":
    main()
