#!/usr/bin/env python3
"""Subscribe to vision/frame/preview and persist decoded frames to disk."""
import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store frames from vision/frame/preview topic")
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--topic", default=os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview"))
    parser.add_argument("--output", default="/mnt/ssd/datasets/snapshots")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames")
    parser.add_argument("--max", type=int, default=0, help="Optional cap on number of frames to save (0=no limit)")
    return parser.parse_args()


def ensure_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir()
    return run_dir


def main():
    args = parse_args()
    output_dir = ensure_run_dir(Path(args.output))
    state = {"last_save": 0.0, "count": 0}

    def on_message(_client: mqtt.Client, _userdata, msg):
        now = time.time()
        if now - state["last_save"] < args.interval:
            return
        state["last_save"] = now
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            return
        b64 = payload.get("image_b64")
        if not b64:
            return
        try:
            data = base64.b64decode(b64)
        except Exception:
            return
        fname = output_dir / f"frame_{state['count']:06d}.jpg"
        with open(fname, "wb") as handle:
            handle.write(data)
        state["count"] += 1
        print(f"saved {fname}")
        if args.max and state["count"] >= args.max:
            _client.loop_stop()

    client = mqtt.Client(client_id="snapshot_logger", protocol=mqtt.MQTTv311)
    client.on_message = on_message
    client.connect(args.host, args.port, 60)
    client.subscribe(args.topic)
    client.loop_start()

    try:
        while True:
            time.sleep(1.0)
            if args.max and state["count"] >= args.max:
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"Saved {state['count']} frames to {output_dir}")


if __name__ == "__main__":
    main()
