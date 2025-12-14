#!/usr/bin/env python3
import json
import os
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
CODEX_CMD_TOPIC = os.getenv("CODEX_CMD_TOPIC", "codex/cmd")
CODEX_REPLY_TOPIC = os.getenv("CODEX_REPLY_TOPIC", "codex/reply")
MEM_TOPIC = os.getenv("MEM_TOPIC", "mem/store")
BUDGET_TOPIC = os.getenv("BUDGET_TOPIC", "budget/update")

queue = []


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe([(CODEX_CMD_TOPIC, 0)])
        client.publish(CODEX_REPLY_TOPIC, json.dumps({"ok": True, "event": "codex_proxy_ready"}))
    else:
        client.publish(CODEX_REPLY_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}))


def summarize_request(task):
    summary = {
        "id": task.get("id") or f"cx_{uuid4().hex[:6]}",
        "intent": task.get("intent", "unknown"),
        "files": task.get("files", []),
        "timestamp": time.time(),
    }
    return summary


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    summary = summarize_request(data if isinstance(data, dict) else {"raw": payload})
    queue.append(summary)
    client.publish(MEM_TOPIC, json.dumps({"op": "append", "key": "codex_tasks", "value": summary}))
    client.publish(BUDGET_TOPIC, json.dumps({"op": "log", "source": "codex_proxy", "tokens": 0}))
    client.publish(
        CODEX_REPLY_TOPIC,
        json.dumps({
            "ok": True,
            "event": "codex_task_recorded",
            "summary": summary,
            "note": "Call Codex manually if required",
        }),
    )


def main():
    client = mqtt.Client(client_id="codex_proxy", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
