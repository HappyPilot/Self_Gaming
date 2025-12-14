#!/usr/bin/env python3
"""Utility to pull YOLO11 weights into the shared /mnt/ssd/models tree."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - script run manually
    raise SystemExit(f"Failed to import ultralytics. Install it first: pip install ultralytics -- reason: {exc}")


DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_OUTPUT = Path("/mnt/ssd/models/yolo")


def resolve_downloaded(model: YOLO, requested: Path) -> Path:
    """Best-effort path resolution for a freshly-downloaded YOLO checkpoint."""
    candidates = []
    for attr in ("ckpt_path", "weights", "model_path"):
        value = getattr(model, attr, None)
        if value:
            candidates.append(Path(value))
    candidates.append(requested)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate downloaded weights for {requested}")


def download(model_name: str, output_dir: Path, force: bool = False) -> Path:
    target_dir = output_dir.expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    requested = Path(model_name)
    yolo = YOLO(model_name)
    downloaded = resolve_downloaded(yolo, requested)
    destination = target_dir / downloaded.name
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return destination
    shutil.copy2(downloaded, destination)
    print(f"[ok] copied {downloaded} -> {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Download YOLO11 weights for the object detection agent")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Checkpoint name or path (default: yolo11n.pt)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Directory to copy weights into")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()
    download(args.model, Path(args.output), force=args.force)


if __name__ == "__main__":
    main()
