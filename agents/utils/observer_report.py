"""Helpers for summarizing agent_observer snapshots."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class SampleReport:
    index: int
    timestamp: float
    snapshot_path: Optional[str]
    snapshot_ok: bool
    action_count: int
    diff_from_prev: Optional[float]
    scene_text: str


def load_image(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def diff_score(img_a: Optional[np.ndarray], img_b: Optional[np.ndarray], size: int = 64) -> Optional[float]:
    if img_a is None or img_b is None:
        return None
    h, w = img_a.shape[:2]
    if h == 0 or w == 0:
        return None
    a = cv2.resize(img_a, (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img_b, (size, size), interpolation=cv2.INTER_AREA)
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(a_gray, b_gray)
    return float(np.mean(diff) / 255.0)


def summarize_samples(samples: List[Dict[str, Any]]) -> List[SampleReport]:
    reports: List[SampleReport] = []
    prev_img = None
    for sample in samples:
        snapshot_path = sample.get("snapshot_path")
        img = load_image(snapshot_path) if snapshot_path else None
        diff = diff_score(prev_img, img)
        prev_img = img if img is not None else prev_img
        scene = sample.get("scene") or {}
        text = scene.get("text") or ""
        if isinstance(text, list):
            text = " ".join(text)
        reports.append(
            SampleReport(
                index=int(sample.get("index", 0)),
                timestamp=float(sample.get("timestamp", 0.0)),
                snapshot_path=snapshot_path,
                snapshot_ok=bool(sample.get("snapshot_ok")),
                action_count=int(sample.get("action_count", 0)),
                diff_from_prev=diff,
                scene_text=str(text)[:120],
            )
        )
    return reports


def progress_summary(reports: List[SampleReport]) -> Dict[str, Any]:
    diffs = [r.diff_from_prev for r in reports if r.diff_from_prev is not None]
    nonzero = [d for d in diffs if d > 0.02]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    return {
        "samples": len(reports),
        "mean_diff": round(mean_diff, 4),
        "nonzero_diffs": len(nonzero),
        "likely_progress": bool(nonzero and len(nonzero) >= max(1, len(diffs) // 2)),
    }


def write_report(output_dir: Path, reports: List[SampleReport]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "summary": progress_summary(reports),
        "samples": [r.__dict__ for r in reports],
    }
    report_json = output_dir / "report.json"
    report_txt = output_dir / "report.txt"
    report_json.write_text(json.dumps(payload, indent=2))
    lines = [
        f"samples={payload['summary']['samples']} mean_diff={payload['summary']['mean_diff']} nonzero={payload['summary']['nonzero_diffs']} likely_progress={payload['summary']['likely_progress']}",
        "",
    ]
    for r in reports:
        lines.append(
            f"idx={r.index} diff={r.diff_from_prev} actions={r.action_count} snapshot={r.snapshot_path} scene={r.scene_text}"
        )
    report_txt.write_text("\n".join(lines))
    return report_json
