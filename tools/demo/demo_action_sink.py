#!/usr/bin/env python3
import json
import os
import signal
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")

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
            print(f"[demo_action_sink] connect rc={rc}, retrying...", flush=True)
        except Exception as exc:
            print(f"[demo_action_sink] connect failed: {exc}. retrying...", flush=True)
        time.sleep(delay)
        delay = min(max_delay, delay * 2)
    return False


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe(ACT_CMD_TOPIC)
        print(f"[demo_action_sink] subscribed to {ACT_CMD_TOPIC}", flush=True)
    else:
        print(f"[demo_action_sink] connect failed rc={rc}", flush=True)


def _on_message(_client, _userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        payload = {"raw": msg.payload.decode("utf-8", "ignore")}
    print(f"[demo_action_sink] {time.time():.3f} {payload}", flush=True)


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client = mqtt.Client(client_id="demo_action_sink", protocol=mqtt.MQTTv311)
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.on_connect = _on_connect
    client.on_message = _on_message
    if not _connect_with_retry(client, MQTT_HOST, MQTT_PORT):
        return
    client.loop_start()

    while not stop:
        time.sleep(0.2)

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
