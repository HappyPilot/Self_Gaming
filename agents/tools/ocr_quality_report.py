#!/usr/bin/env python3
"""Capture OCR samples and optionally save frames for quality review."""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from PIL import Image

from utils.frame_transport import get_frame_bytes


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


class OcrReporter:
    def __init__(
        self,
        host: str,
        port: int,
        frame_topic: str,
        ocr_topic: str,
        simple_topic: str,
        ocr_cmd_topic: str,
        out_path: Path,
        save_frames: bool,
        samples: int,
        require_text: bool,
    ):
        self.host = host
        self.port = port
        self.frame_topic = frame_topic
        self.ocr_topic = ocr_topic
        self.simple_topic = simple_topic
        self.ocr_cmd_topic = ocr_cmd_topic
        self.out_path = out_path
        self.save_frames = save_frames
        self.samples = max(1, samples)
        self.require_text = require_text

        self.client = mqtt.Client(client_id=f"ocr_quality_{os.getpid()}", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._last_frame_meta = {}
        self._stop = threading.Event()
        self._count = 0
        self._queue: "queue.Queue[dict]" = queue.Queue()

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) != 0:
            raise RuntimeError(f"MQTT connect failed rc={rc}")
        topics = [(self.frame_topic, 0), (self.ocr_topic, 0), (self.simple_topic, 0)]
        client.subscribe(topics)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            payload = {}
        if msg.topic == self.frame_topic:
            frame_bytes = get_frame_bytes(payload)
            if not frame_bytes:
                return
            with self._frame_lock:
                self._last_frame = frame_bytes
                self._last_frame_meta = payload
            return
        if msg.topic == self.ocr_topic:
            entry = self._build_entry("ocr_easy", payload)
        elif msg.topic == self.simple_topic:
            entry = self._build_entry("simple_ocr", payload)
        else:
            return
        if entry is None:
            return
        self._queue.put(entry)

    def _build_entry(self, source: str, payload: dict) -> dict | None:
        if not isinstance(payload, dict):
            return None
        ok = payload.get("ok", True)
        if ok is False:
            return {
                "ts": time.time(),
                "source": source,
                "ok": False,
                "error": payload.get("error"),
                "raw": payload,
            }
        if source == "ocr_easy":
            text = str(payload.get("text") or "").strip()
            results = payload.get("results") or []
            items = [
                {"text": r.get("text"), "conf": r.get("conf"), "box": r.get("box")}
                for r in results
                if isinstance(r, dict)
            ]
        else:
            items = payload.get("items") or []
            text = "\n".join([str(it.get("text") or "").strip() for it in items if it.get("text")])
        if self.require_text and not text:
            return None
        entry = {
            "ts": time.time(),
            "source": source,
            "ok": True,
            "text": text,
            "items": items,
            "count": len(items),
        }
        if self.save_frames:
            frame_path = self._save_frame()
            if frame_path:
                entry["frame"] = str(frame_path)
        return entry

    def _save_frame(self) -> Path | None:
        with self._frame_lock:
            frame_bytes = self._last_frame
        if not frame_bytes:
            return None
        try:
            img = Image.open(io.BytesIO(frame_bytes))
            ts = int(time.time() * 1000)
            frame_dir = self.out_path.parent / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            path = frame_dir / f"frame_{ts}.jpg"
            img.save(path, format="JPEG", quality=95)
            return path
        except Exception:
            return None

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.connect(self.host, self.port, 30)
        self.client.loop_start()
        with self.out_path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set() and self._count < self.samples:
                try:
                    entry = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
                handle.flush()
                if entry.get("ok") or not self.require_text:
                    self._count += 1
        self._stop.set()
        self.client.loop_stop()
        self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Capture OCR samples for quality review.")
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--frame-topic", default=os.getenv("OCR_FRAME_TOPIC", "vision/frame/full"))
    parser.add_argument("--ocr-topic", default=os.getenv("OCR_EASY_TOPIC", "ocr_easy/text"))
    parser.add_argument("--simple-topic", default=os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text"))
    parser.add_argument("--ocr-cmd-topic", default=os.getenv("OCR_CMD", "ocr_easy/cmd"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--out", default="logs/ocr_report/ocr_report.jsonl")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--request", action="store_true")
    parser.add_argument("--require-text", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    reporter = OcrReporter(
        host=args.host,
        port=args.port,
        frame_topic=args.frame_topic,
        ocr_topic=args.ocr_topic,
        simple_topic=args.simple_topic,
        ocr_cmd_topic=args.ocr_cmd_topic,
        out_path=out_path,
        save_frames=args.save_frames,
        samples=args.samples,
        require_text=args.require_text,
    )

    request_thread = None
    if args.request:
        def _request_loop():
            while not reporter._stop.is_set():
                reporter.client.publish(args.ocr_cmd_topic, json.dumps({"cmd": "once", "timeout": args.interval}))
                reporter._stop.wait(args.interval)
        request_thread = threading.Thread(target=_request_loop, daemon=True)
        request_thread.start()

    reporter.start()
    if request_thread:
        reporter._stop.set()
        request_thread.join(timeout=2)


if __name__ == "__main__":
    main()
