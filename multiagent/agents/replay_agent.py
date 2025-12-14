#!/usr/bin/env python3
"""Replay buffer agent that stores transitions and serves prioritized samples."""
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
STORE_TOPIC = os.getenv("REPLAY_STORE_TOPIC", "replay/store")
SAMPLE_REQ_TOPIC = os.getenv("REPLAY_SAMPLE_TOPIC", "replay/sample-request")
SAMPLE_RESP_TOPIC = os.getenv("REPLAY_SAMPLE_RESP_TOPIC", "replay/sample-response")
BUFFER_SIZE = int(os.getenv("REPLAY_BUFFER_SIZE", "5000"))
MIN_PRIORITY = float(os.getenv("REPLAY_MIN_PRIORITY", "1e-3"))

buffer = deque(maxlen=BUFFER_SIZE)
priority_sum = 0.0


def add_transition(data):
    global priority_sum
    priority = float(data.get("priority", 1.0))
    priority = max(priority, MIN_PRIORITY)
    transition = {
        "timestamp": data.get("timestamp", time.time()),
        "state": data.get("state"),
        "next_state": data.get("next_state"),
        "action": data.get("action"),
        "reward": data.get("reward"),
        "teacher": data.get("teacher"),
        "task": data.get("task"),
        "extra": data.get("extra"),
        "priority": priority,
    }
    if len(buffer) == BUFFER_SIZE:
        popped = buffer.popleft()
        priority_sum -= popped.get("priority", 0.0)
    buffer.append(transition)
    priority_sum += priority


def sample_transitions(count: int):
    if not buffer:
        return []
    if priority_sum <= 0:
        return random.sample(list(buffer), min(count, len(buffer)))
    samples = []
    # simple roulette-wheel sampling
    for _ in range(min(count, len(buffer))):
        target = random.uniform(0, priority_sum)
        cumulative = 0.0
        chosen = buffer[-1]
        for transition in buffer:
            cumulative += transition.get("priority", MIN_PRIORITY)
            if cumulative >= target:
                chosen = transition
                break
        samples.append(chosen)
    return samples


def on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe([(STORE_TOPIC, 0), (SAMPLE_REQ_TOPIC, 0)])
        client.publish(
            SAMPLE_RESP_TOPIC,
            json.dumps({"ok": True, "event": "replay_ready", "size": len(buffer)}),
        )
    else:
        client.publish(
            SAMPLE_RESP_TOPIC,
            json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}),
        )


def on_message(client, _userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    if msg.topic == STORE_TOPIC:
        add_transition(data)
    elif msg.topic == SAMPLE_REQ_TOPIC:
        count = int(data.get("count", 32))
        request_id = data.get("request_id")
        samples = sample_transitions(count)
        client.publish(
            SAMPLE_RESP_TOPIC,
            json.dumps(
                {
                    "ok": True,
                    "event": "replay_samples",
                    "request_id": request_id,
                    "samples": samples,
                    "timestamp": time.time(),
                }
            ),
        )


def main():
    client = mqtt.Client(client_id="replay_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
