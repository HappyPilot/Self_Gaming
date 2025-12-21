#!/usr/bin/env python3
"""Export Ultralytics YOLO weights to ONNX."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - script run manually
    raise SystemExit(f"Failed to import ultralytics. Install it first: pip install ultralytics -- reason: {exc}")


def resolve_export_path(exported, search_dirs: list[Path]) -> Path:
    candidates: list[Path] = []
    if isinstance(exported, (str, Path)):
        candidates.append(Path(exported))
    elif isinstance(exported, (list, tuple)):
        for item in exported:
            if isinstance(item, (str, Path)):
                candidates.append(Path(item))
    elif isinstance(exported, dict):
        for key in ("file", "path", "model"):
            value = exported.get(key)
            if value:
                candidates.append(Path(value))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for directory in search_dirs:
        if not directory.exists():
            continue
        newest = max(directory.glob("*.onnx"), key=lambda p: p.stat().st_mtime, default=None)
        if newest:
            return newest

    raise FileNotFoundError("Could not locate exported ONNX file")


def export_onnx(
    weights: Path,
    output: Path | None,
    imgsz: int,
    opset: int,
    dynamic: bool,
    half: bool,
    device: str,
    force: bool,
) -> Path:
    weights = weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        device=device,
        half=half,
    )
    export_path = resolve_export_path(exported, [weights.parent, Path.cwd()])

    if output is None:
        return export_path

    output = output.expanduser()
    if output.suffix.lower() != ".onnx":
        output.mkdir(parents=True, exist_ok=True)
        output = output / export_path.name

    if output.exists() and not force:
        print(f"[skip] {output} already exists (use --force to overwrite)")
        return output

    if output.resolve() == export_path.resolve():
        print(f"[ok] export ready: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(export_path, output)
    print(f"[ok] copied {export_path} -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX")
    parser.add_argument("weights", help="Path to YOLO weights (.pt)")
    parser.add_argument("--output", help="Output .onnx path or directory (default: same dir as weights)")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (default: 640)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    parser.add_argument("--half", action="store_true", help="Export in FP16 when supported")
    parser.add_argument("--device", default="cpu", help="Export device (cpu, cuda:0, etc)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    export_onnx(
        Path(args.weights),
        output,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
