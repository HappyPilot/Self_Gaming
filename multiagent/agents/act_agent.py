#!/usr/bin/env python3
import json
import os
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACT_CMD_ALIAS = os.getenv("ACT_CMD_ALIAS", "act/request")
POLICY_ACTION_TOPIC = os.getenv("POLICY_ACTION_TOPIC", "control/keys")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
ACT_RESULT_ALIAS = os.getenv("ACT_RESULT_ALIAS", "act/feedback")

last_action = {}


def _publish_act(client, payload):
    for topic in {ACT_RESULT_TOPIC, ACT_RESULT_ALIAS}:
        if topic:
            client.publish(topic, json.dumps(payload))


def on_connect(client, userdata, flags, rc):
    topics = {(ACT_CMD_TOPIC, 0), (POLICY_ACTION_TOPIC, 0)}
    if ACT_CMD_ALIAS and ACT_CMD_ALIAS != ACT_CMD_TOPIC:
        topics.add((ACT_CMD_ALIAS, 0))
    if rc == 0:
        client.subscribe(list(topics))
        _publish_act(client, {"ok": True, "event": "act_agent_ready"})
    else:
        _publish_act(client, {"ok": False, "event": "connect_failed", "code": int(rc)})


def apply_action(action):
    # Placeholder for real device interactions
    time.sleep(0.1)
    return {"ok": True, "applied": action, "timestamp": time.time()}


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    if msg.topic == POLICY_ACTION_TOPIC:
        last_action.update(data if isinstance(data, dict) else {"raw": payload})
    elif msg.topic in (ACT_CMD_TOPIC, ACT_CMD_ALIAS):
        action = data.get("action") if isinstance(data, dict) else last_action
        if not action:
            _publish_act(client, {"ok": False, "error": "no_action"})
            return
        result = apply_action(action)
        _publish_act(client, result)


def main():
    client = mqtt.Client(client_id="act_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
