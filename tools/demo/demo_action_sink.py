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
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_start()

    while not stop:
        time.sleep(0.2)

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
