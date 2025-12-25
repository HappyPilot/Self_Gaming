"""Prepare a simple VLA dataset from recorder sessions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from training.vla.utils import iter_jsonl, write_jsonl


def _parse_frame_ts(path: Path) -> Optional[float]:
    stem = path.stem
    if not stem:
        return None
    prefix = stem.split("_")[0]
    try:
        return int(prefix) / 1000.0
    except ValueError:
        return None


def _load_actions(path: Path) -> List[Dict]:
    actions: List[Dict] = []
    if not path.exists():
        return actions
    for row in iter_jsonl(path):
        try:
            ts = float(row.get("timestamp"))
        except (TypeError, ValueError):
            continue
        action = row.get("action")
        if not isinstance(action, dict):
            continue
        actions.append({"timestamp": ts, "topic": row.get("topic"), "action": action})
    actions.sort(key=lambda item: item["timestamp"])
    return actions


def _load_sensors(path: Path) -> List[Dict]:
    sensors: List[Dict] = []
    if not path.exists():
        return sensors
    for row in iter_jsonl(path):
        try:
            ts = float(row.get("timestamp"))
        except (TypeError, ValueError):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        sensors.append({"timestamp": ts, "topic": row.get("topic"), "payload": payload})
    sensors.sort(key=lambda item: item["timestamp"])
    return sensors


def _list_sessions(dataset_dir: Path, game_id: Optional[str]) -> Iterable[Tuple[Path, str]]:
    if game_id:
        game_dirs = [dataset_dir / game_id]
    else:
        game_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    for game_dir in game_dirs:
        if not game_dir.exists():
            continue
        for session_dir in sorted(game_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            frames_dir = session_dir / "frames"
            actions_path = session_dir / "actions.jsonl"
            if frames_dir.is_dir() and actions_path.exists():
                yield session_dir, game_dir.name


def _latest_sensors(
    sensors: List[Dict],
    frame_ts: float,
    sensor_idx: int,
    latest_by_topic: Dict[str, Dict],
    latest_ts: Dict[str, float],
) -> int:
    while sensor_idx < len(sensors) and sensors[sensor_idx]["timestamp"] <= frame_ts:
        row = sensors[sensor_idx]
        topic = row.get("topic")
        if isinstance(topic, str):
            latest_by_topic[topic] = row["payload"]
            latest_ts[topic] = row["timestamp"]
        sensor_idx += 1
    return sensor_idx


def build_samples(
    session_dir: Path,
    horizon_sec: float,
    sensor_window_sec: float,
    sensor_topics: Optional[List[str]],
) -> List[Dict]:
    frames_dir = session_dir / "frames"
    actions_path = session_dir / "actions.jsonl"
    sensors_path = session_dir / "sensors.jsonl"
    meta_path = session_dir / "meta.json"

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    actions = _load_actions(actions_path)
    sensors = _load_sensors(sensors_path)
    frames = []
    for frame_path in sorted(frames_dir.glob("*.jpg")):
        ts = _parse_frame_ts(frame_path)
        if ts is None:
            continue
        frames.append((ts, frame_path))

    samples: List[Dict] = []
    action_idx = 0
    sensor_idx = 0
    latest_by_topic: Dict[str, Dict] = {}
    latest_ts: Dict[str, float] = {}

    for frame_ts, frame_path in frames:
        while action_idx < len(actions) and actions[action_idx]["timestamp"] < frame_ts:
            action_idx += 1
        end_ts = frame_ts + horizon_sec
        chunk = []
        j = action_idx
        while j < len(actions) and actions[j]["timestamp"] <= end_ts:
            chunk.append(actions[j])
            j += 1

        sensor_idx = _latest_sensors(sensors, frame_ts, sensor_idx, latest_by_topic, latest_ts)
        sensors_out: Dict[str, Dict] = {}
        for topic, payload in latest_by_topic.items():
            if sensor_topics and topic not in sensor_topics:
                continue
            ts = latest_ts.get(topic)
            if ts is None:
                continue
            if sensor_window_sec <= 0 or (frame_ts - ts) <= sensor_window_sec:
                sensors_out[topic] = payload

        sample = {
            "frame_path": str(frame_path),
            "frame_ts": frame_ts,
            "actions": chunk,
            "sensors": sensors_out,
            "session_id": meta.get("session_id", session_dir.name),
            "game_id": meta.get("game_id", session_dir.parent.name),
            "profile": meta.get("profile"),
        }
        samples.append(sample)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VLA dataset from recorder sessions")
    parser.add_argument("--dataset-dir", default=os.getenv("RECORDER_DATASET_DIR", "/mnt/ssd/datasets"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--horizon-sec", type=float, default=0.5)
    parser.add_argument("--sensor-window-sec", type=float, default=0.5)
    parser.add_argument("--sensor-topics", default="scene/state,vision/observation,vision/objects")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    sensor_topics = [topic.strip() for topic in args.sensor_topics.split(",") if topic.strip()]

    all_samples: List[Dict] = []
    for session_dir, _game_id in _list_sessions(dataset_dir, args.game_id):
        samples = build_samples(
            session_dir=session_dir,
            horizon_sec=args.horizon_sec,
            sensor_window_sec=args.sensor_window_sec,
            sensor_topics=sensor_topics or None,
        )
        all_samples.extend(samples)
        if args.max_samples and len(all_samples) >= args.max_samples:
            all_samples = all_samples[: args.max_samples]
            break

    if not all_samples:
        raise SystemExit("No samples found. Check dataset-dir, game-id, or recorder output.")

    write_jsonl(output_path, all_samples)
    print(f"Wrote {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
