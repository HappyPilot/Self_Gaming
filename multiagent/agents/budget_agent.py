#!/usr/bin/env python3
import json
import os
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
BUDGET_TOPIC = os.getenv("BUDGET_TOPIC", "budget/update")
BUDGET_SUMMARY_TOPIC = os.getenv("BUDGET_SUMMARY_TOPIC", "budget/summary")
TOKEN_LIMIT = int(os.getenv("BUDGET_TOKEN_LIMIT", "20000"))

state = {
    "tokens": 0,
    "gpu_hours": 0.0,
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe([(BUDGET_TOPIC, 0)])
        client.publish(
            BUDGET_SUMMARY_TOPIC,
            json.dumps({"ok": True, "event": "budget_agent_ready", "limit": TOKEN_LIMIT}),
        )
    else:
        client.publish(BUDGET_SUMMARY_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    state["tokens"] += int(data.get("tokens", 0))
    state["gpu_hours"] += float(data.get("gpu_hours", 0.0))
    summary = {
        "ok": True,
        "event": "budget_update",
        "tokens": state["tokens"],
        "gpu_hours": round(state["gpu_hours"], 3),
        "limit": TOKEN_LIMIT,
        "timestamp": time.time(),
        "throttle": state["tokens"] >= TOKEN_LIMIT,
    }
    client.publish(BUDGET_SUMMARY_TOPIC, json.dumps(summary))


def main():
    client = mqtt.Client(client_id="budget_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
