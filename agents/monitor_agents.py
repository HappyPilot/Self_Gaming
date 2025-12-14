#!/usr/bin/env python3
"""Simple MQTT monitor for demonstrator and auto-trainer agents.

The script subscribes to status topics and prints compact log entries with
timestamps so you can keep an eye on automated action generation and training
triggers. Press Ctrl+C to exit.
"""
import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Iterable, Tuple

import paho.mqtt.client as mqtt

DEFAULT_TOPICS: Tuple[str, ...] = (
    "demonstrator/status",
    "auto_train/status",
    "train/status",
)

stop_event = threading.Event()


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def on_connect(client: mqtt.Client, _userdata, _flags, rc, _properties=None):
    if rc == 0:
        topics = [(topic, 0) for topic in client._userdata["topics"]]
        client.subscribe(topics)
        print(f"[{_timestamp()}] connected -> {', '.join(t for t, _ in topics)}", flush=True)
    else:
        print(f"[{_timestamp()}] connect_failed rc={rc}", file=sys.stderr, flush=True)


def on_message(_client: mqtt.Client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8", "ignore")
        data = json.loads(payload)
    except Exception:
        data = payload
    summary = data
    if isinstance(data, dict):
        summary = {
            key: data.get(key)
            for key in ("event", "ok", "job_id", "samples", "action", "error", "score")
            if key in data
        }
    print(f"[{_timestamp()}] {msg.topic}: {summary}", flush=True)


def on_disconnect(_client: mqtt.Client, _userdata, rc, _properties=None):
    print(f"[{_timestamp()}] disconnected rc={rc}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor agent status topics over MQTT")
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--topics",
        default=",".join(DEFAULT_TOPICS),
        help="Comma-separated list of topics to subscribe to",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    topics = tuple(t.strip() for t in args.topics.split(",") if t.strip()) or DEFAULT_TOPICS

    client = mqtt.Client(
        client_id="agent_monitor",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        userdata={"topics": topics},
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    def handle_signal(_signum, _frame):
        stop_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        client.connect(args.host, args.port, keepalive=30)
    except Exception as exc:
        print(f"[{_timestamp()}] connection error: {exc}", file=sys.stderr, flush=True)
        return 1

    client.loop_start()
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        client.loop_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
