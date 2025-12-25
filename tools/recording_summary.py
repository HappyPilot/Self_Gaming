#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"warning: failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: tools/recording_summary.py <session_dir>", file=sys.stderr)
        return 1

    session_dir = Path(sys.argv[1])
    if not session_dir.exists() or not session_dir.is_dir():
        print(f"error: session directory not found: {session_dir}", file=sys.stderr)
        return 1

    meta = _load_json(session_dir / "meta.json") or {}
    qc = _load_json(session_dir / "qc.json") or {}
    frames_dir = session_dir / "frames"
    frames_count = sum(1 for _ in frames_dir.glob("*.jpg")) if frames_dir.exists() else 0
    actions_count = _count_lines(session_dir / "actions.jsonl")
    sensors_count = _count_lines(session_dir / "sensors.jsonl")

    print(f"session_dir: {session_dir}")
    if meta:
        print(f"game_id: {meta.get('game_id')}")
        print(f"session_id: {meta.get('session_id')}")
        print(f"started_at: {meta.get('started_at')}")
        print(f"frame_topic: {meta.get('frame_topic')}")
        print(f"expected_fps: {meta.get('expected_fps')}")

    print(f"frames: {frames_count}")
    print(f"actions: {actions_count}")
    print(f"sensors: {sensors_count}")

    if qc:
        print("qc:")
        for key in sorted(qc.keys()):
            print(f"  {key}: {qc[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
