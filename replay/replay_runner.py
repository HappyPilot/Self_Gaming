#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import paho.mqtt.client as mqtt
except Exception:  # noqa: BLE001
    mqtt = None


LOG_LEVEL = os.getenv("REPLAY_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("replay_runner")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "30"))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "replay_runner")

REPLAY_SPEED = os.getenv("REPLAY_SPEED", "1.0")
REPLAY_MAX_SEC = os.getenv("REPLAY_MAX_SEC", "0")
REPLAY_START_DELAY_SEC = os.getenv("REPLAY_START_DELAY_SEC", "0.5")
REPLAY_PUBLISH_FRAMES = os.getenv("REPLAY_PUBLISH_FRAMES", "1") != "0"
REPLAY_PUBLISH_SENSORS = os.getenv("REPLAY_PUBLISH_SENSORS", "1") != "0"
REPLAY_SENSOR_TOPICS = os.getenv("REPLAY_SENSOR_TOPICS", "")
REPLAY_FRAME_TOPIC = os.getenv("REPLAY_FRAME_TOPIC", "")


def _env_float(raw: str, default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_topics(raw: str) -> Optional[set[str]]:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}


def _parse_frame_timestamp(path: Path) -> float:
    stem = path.stem
    token = stem.split("_", 1)[0]
    try:
        return int(token) / 1000.0
    except ValueError:
        try:
            return path.stat().st_mtime
        except OSError:
            return time.time()


_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _jpeg_dimensions(data: bytes) -> tuple[Optional[int], Optional[int]]:
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None, None
    i = 2
    length = len(data)
    while i < length:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < length and data[i] == 0xFF:
            i += 1
        if i >= length:
            break
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9):
            continue
        if i + 1 >= length:
            break
        segment_length = (data[i] << 8) | data[i + 1]
        if segment_length < 2:
            break
        if i + segment_length > length:
            break
        segment_start = i + 2
        if marker in _JPEG_SOF_MARKERS:
            if segment_start + 4 >= length:
                break
            height = (data[segment_start + 1] << 8) | data[segment_start + 2]
            width = (data[segment_start + 3] << 8) | data[segment_start + 4]
            return width, height
        i += segment_length
    return None, None


def _variant_for_topic(topic: str) -> str:
    if topic.endswith("/full"):
        return "full"
    if topic.endswith("/preview"):
        return "preview"
    return "replay"


@dataclass(frozen=True)
class ReplayEvent:
    ts: float
    order: int
    kind: str
    topic: str
    payload: object


@dataclass
class ReplayStats:
    events_total: int = 0
    frames_published: int = 0
    sensors_published: int = 0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    duration_sec: float = 0.0


def _iter_frame_events(frames_dir: Path, topic: str) -> Iterable[ReplayEvent]:
    order = 0
    for path in sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg")):
        ts = _parse_frame_timestamp(path)
        yield ReplayEvent(ts=ts, order=order, kind="frame", topic=topic, payload=path)
        order += 1


def _iter_sensor_events(path: Path, allowed_topics: Optional[set[str]]) -> Iterable[ReplayEvent]:
    order = 0
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in %s:%s", path, line_no)
                continue
            if not isinstance(data, dict):
                continue
            ts = data.get("timestamp")
            topic = data.get("topic")
            payload = data.get("payload")
            if ts is None or not topic:
                continue
            if allowed_topics and topic not in allowed_topics:
                continue
            try:
                ts_value = float(ts)
            except (TypeError, ValueError):
                continue
            yield ReplayEvent(ts=ts_value, order=order, kind="sensor", topic=str(topic), payload=payload)
            order += 1


def build_events(
    session_dir: Path,
    *,
    include_frames: bool,
    include_sensors: bool,
    frame_topic: str,
    sensor_topics: Optional[set[str]],
) -> list[ReplayEvent]:
    events: list[ReplayEvent] = []
    if include_frames and frame_topic:
        frames_dir = session_dir / "frames"
        events.extend(_iter_frame_events(frames_dir, frame_topic))
    if include_sensors:
        sensors_path = session_dir / "sensors.jsonl"
        events.extend(_iter_sensor_events(sensors_path, sensor_topics))
    events.sort(key=lambda item: (item.ts, item.order))
    return events


class ReplayRunner:
    def __init__(
        self,
        session_dir: Path,
        *,
        speed: float = 1.0,
        max_sec: float = 0.0,
        start_delay: float = 0.5,
        include_frames: bool = True,
        include_sensors: bool = True,
        frame_topic: Optional[str] = None,
        sensor_topics: Optional[set[str]] = None,
        dry_run: bool = False,
    ) -> None:
        self.session_dir = session_dir
        self.speed = speed if speed > 0 else 1.0
        self.max_sec = max_sec
        self.start_delay = max(0.0, start_delay)
        self.include_frames = include_frames
        self.include_sensors = include_sensors
        self.sensor_topics = sensor_topics
        self.dry_run = dry_run
        self.meta = _load_json(session_dir / "meta.json")
        self.frame_topic = (
            frame_topic
            or REPLAY_FRAME_TOPIC
            or self.meta.get("frame_topic")
            or os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
        )
        self.client = None
        if not dry_run:
            if mqtt is None:
                raise SystemExit("paho-mqtt is not installed; cannot publish replay")
            self.client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)

    def _connect(self) -> None:
        if not self.client:
            return
        self.client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
        self.client.loop_start()

    def _disconnect(self) -> None:
        if not self.client:
            return
        self.client.loop_stop()
        self.client.disconnect()

    def _publish_frame(self, topic: str, ts: float, path: Path) -> bool:
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read frame %s: %s", path, exc)
            return False
        width, height = _jpeg_dimensions(data)
        if width is None or height is None:
            logger.debug("Failed to parse JPEG dimensions for %s", path)
            width = height = 0
        variant = _variant_for_topic(topic)
        # Keep keys aligned with docs/mqtt/topics.md and utils.frame_transport.get_frame_bytes.
        payload = {
            "ok": True,
            "timestamp": ts,
            "width": width,
            "height": height,
            "variant": variant,
            "image_b64": base64.b64encode(data).decode("ascii"),
        }
        try:
            self.client.publish(topic, json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish frame: %s", exc)
            return False
        return True

    def _publish_sensor(self, topic: str, payload: object) -> bool:
        try:
            packed = json.dumps(payload)
        except TypeError as exc:
            logger.warning("Sensor payload not JSON-serializable for %s: %s", topic, exc)
            return False
        try:
            self.client.publish(topic, packed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish sensor: %s", exc)
            return False
        return True

    def run(self) -> ReplayStats:
        stats = ReplayStats()
        events = build_events(
            self.session_dir,
            include_frames=self.include_frames,
            include_sensors=self.include_sensors,
            frame_topic=self.frame_topic,
            sensor_topics=self.sensor_topics,
        )
        stats.events_total = len(events)
        if not events:
            logger.warning("No replay events found in %s", self.session_dir)
            return stats

        start_ts = events[0].ts
        end_ts = events[-1].ts
        stats.start_ts = start_ts
        stats.end_ts = end_ts
        stats.duration_sec = max(0.0, end_ts - start_ts)

        logger.info(
            "Replay start: events=%s duration=%.2fs speed=%.2fx",
            stats.events_total,
            stats.duration_sec,
            self.speed,
        )

        if not self.dry_run:
            self._connect()
        start_wall = time.time() + self.start_delay

        try:
            for event in events:
                elapsed = event.ts - start_ts
                if self.max_sec > 0 and elapsed > self.max_sec:
                    break
                target = start_wall + (elapsed / self.speed)
                sleep_for = target - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                if event.kind == "frame":
                    if self.dry_run:
                        stats.frames_published += 1
                    elif self._publish_frame(event.topic, event.ts, event.payload):  # type: ignore[arg-type]
                        stats.frames_published += 1
                elif event.kind == "sensor":
                    if self.dry_run:
                        stats.sensors_published += 1
                    elif self._publish_sensor(event.topic, event.payload):
                        stats.sensors_published += 1
        finally:
            if not self.dry_run:
                self._disconnect()

        logger.info(
            "Replay done: frames=%s sensors=%s",
            stats.frames_published,
            stats.sensors_published,
        )
        return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a recorded session via MQTT")
    parser.add_argument("session_dir", nargs="?", help="Path to session directory")
    parser.add_argument("--speed", type=float, default=_env_float(REPLAY_SPEED, 1.0))
    parser.add_argument("--max-sec", type=float, default=_env_float(REPLAY_MAX_SEC, 0.0))
    parser.add_argument("--start-delay", type=float, default=_env_float(REPLAY_START_DELAY_SEC, 0.5))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--no-sensors", action="store_true")
    parser.add_argument("--frame-topic", default=REPLAY_FRAME_TOPIC)
    parser.add_argument("--sensor-topics", default=REPLAY_SENSOR_TOPICS)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    session_dir = args.session_dir or os.getenv("REPLAY_SESSION_DIR")
    if not session_dir:
        parser.print_usage()
        return 2
    runner = ReplayRunner(
        Path(session_dir),
        speed=args.speed,
        max_sec=args.max_sec,
        start_delay=args.start_delay,
        include_frames=not args.no_frames and REPLAY_PUBLISH_FRAMES,
        include_sensors=not args.no_sensors and REPLAY_PUBLISH_SENSORS,
        frame_topic=args.frame_topic or None,
        sensor_topics=_parse_topics(args.sensor_topics),
        dry_run=args.dry_run,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
