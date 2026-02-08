#!/usr/bin/env python3
"""Evaluate observer samples against HF vljepa probe datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.utils import hf_vljepa_probe_eval as probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Score observer samples against HF vljepa probe datasets")
    parser.add_argument("--samples", required=True, help="Path to agent_observer samples.json")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="HF dataset id (repeatable). Default: 4 ClarusC64 vljepa probes.",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    samples = probe.load_samples(samples_path)
    result = probe.run_probe_eval(samples, dataset_ids=args.dataset or None)
    payload = {
        "samples_path": str(samples_path),
        "sample_count": len(samples),
        **result,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
