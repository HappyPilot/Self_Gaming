#!/usr/bin/env python3
import json
import os
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EVAL_CMD_TOPIC = os.getenv("EVAL_CMD_TOPIC", "eval/cmd")
EVAL_CMD_ALIAS = os.getenv("EVAL_CMD_ALIAS", "eval/request")
EVAL_RESULT_TOPIC = os.getenv("EVAL_RESULT_TOPIC", "eval/result")
EVAL_RESULT_ALIAS = os.getenv("EVAL_RESULT_ALIAS", "eval/report")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")

last_scene = {}


def _publish_eval(client, payload):
    for topic in {EVAL_RESULT_TOPIC, EVAL_RESULT_ALIAS}:
        if topic:
            client.publish(topic, json.dumps(payload))


def on_connect(client, userdata, flags, rc):
    topics = {(EVAL_CMD_TOPIC, 0), (SCENE_TOPIC, 0)}
    if EVAL_CMD_ALIAS and EVAL_CMD_ALIAS != EVAL_CMD_TOPIC:
        topics.add((EVAL_CMD_ALIAS, 0))
    if rc == 0:
        client.subscribe(list(topics))
        _publish_eval(client, {"ok": True, "event": "eval_agent_ready"})
    else:
        _publish_eval(client, {"ok": False, "event": "connect_failed", "code": int(rc)})


def run_eval(plan, scene):
    time.sleep(0.5)
    return {
        "ok": True,
        "plan": plan,
        "scene": scene,
        "score": round(0.5 + 0.5 * (hash(plan) % 10) / 10, 2),
        "timestamp": time.time(),
    }


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    if msg.topic == SCENE_TOPIC and data.get("ok"):
        last_scene.update(data)
    elif msg.topic in (EVAL_CMD_TOPIC, EVAL_CMD_ALIAS):
        plan_id = data.get("plan_id") if isinstance(data, dict) else f"plan_{uuid4().hex[:6]}"
        result = run_eval(plan_id, last_scene)
        _publish_eval(client, result)


def main():
    client = mqtt.Client(client_id="eval_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
