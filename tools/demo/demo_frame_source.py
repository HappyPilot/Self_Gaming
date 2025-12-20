#!/usr/bin/env python3
import json
import math
import os
import signal
import time

import paho.mqtt.client as mqtt

DEFAULT_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
)

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("FRAME_TOPIC", "vision/frame/preview")
MEAN_TOPIC = os.getenv("MEAN_TOPIC", "vision/mean")
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "1.0"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
IMAGE_B64 = os.getenv("DEMO_IMAGE_B64", DEFAULT_IMAGE_B64).strip()
IMAGE_WIDTH = int(os.getenv("DEMO_IMAGE_WIDTH", "1"))
IMAGE_HEIGHT = int(os.getenv("DEMO_IMAGE_HEIGHT", "1"))

stop = False


def _handle_signal(_signum, _frame):
    global stop
    stop = True


def _connect_with_retry(client, host: str, port: int) -> bool:
    delay = 0.5
    max_delay = 5.0
    while not stop:
        try:
            rc = client.connect(host, port, 30)
            if rc == 0:
                return True
            print(f"[demo_frame_source] connect rc={rc}, retrying...", flush=True)
        except Exception as exc:
            print(f"[demo_frame_source] connect failed: {exc}. retrying...", flush=True)
        time.sleep(delay)
        delay = min(max_delay, delay * 2)
    return False


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client = mqtt.Client(client_id="demo_frame_source", protocol=mqtt.MQTTv311)
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    if not _connect_with_retry(client, MQTT_HOST, MQTT_PORT):
        return
    client.loop_start()

    tick = 0
    print(
        f"[demo_frame_source] publishing {FRAME_TOPIC} and {MEAN_TOPIC} to {MQTT_HOST}:{MQTT_PORT}",
        flush=True,
    )
    try:
        while not stop:
            ts = time.time()
            mean = 0.5 + 0.4 * math.sin(tick / 6.0)
            frame_payload = {
                "ok": True,
                "timestamp": ts,
                "image_b64": IMAGE_B64,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "source": "demo_frame_source",
            }
            mean_payload = {
                "ok": True,
                "mean": round(mean, 3),
                "timestamp": ts,
                "source": "demo_frame_source",
            }
            client.publish(FRAME_TOPIC, json.dumps(frame_payload))
            client.publish(MEAN_TOPIC, json.dumps(mean_payload))
            if LOG_EVERY > 0 and tick % LOG_EVERY == 0:
                print(f"[demo_frame_source] tick={tick} mean={mean_payload['mean']}", flush=True)
            tick += 1
            time.sleep(FRAME_INTERVAL)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
