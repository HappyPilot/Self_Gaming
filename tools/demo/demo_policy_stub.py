#!/usr/bin/env python3
import json
import os
import signal
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACTION_SEQUENCE = [
    action.strip()
    for action in os.getenv("DEMO_ACTIONS", "noop,click,move,key").split(",")
    if action.strip()
]
MIN_INTERVAL = float(os.getenv("DEMO_ACTION_MIN_INTERVAL", "0.5"))

if not ACTION_SEQUENCE:
    ACTION_SEQUENCE = ["noop"]

stop = False
state = {"last_action_ts": 0.0, "action_index": 0}


def _handle_signal(_signum, _frame):
    global stop
    stop = True


def _build_action(name: str) -> dict:
    if name == "click":
        return {"action": "click", "button": "left"}
    if name == "move":
        return {"action": "mouse_move", "dx": 12, "dy": 0}
    if name in {"key", "keypress"}:
        return {"action": "key_press", "key": "space"}
    return {"action": name}


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe(SCENE_TOPIC)
        print(f"[demo_policy_stub] subscribed to {SCENE_TOPIC}", flush=True)
    else:
        print(f"[demo_policy_stub] connect failed rc={rc}", flush=True)


def _on_message(client, _userdata, msg):
    now = time.time()
    if now - state["last_action_ts"] < MIN_INTERVAL:
        return
    try:
        scene = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        scene = {}
    action_name = ACTION_SEQUENCE[state["action_index"] % len(ACTION_SEQUENCE)]
    payload = _build_action(action_name)
    payload.update(
        {
            "ok": True,
            "source": "demo_policy_stub",
            "timestamp": now,
        }
    )
    if isinstance(scene, dict):
        if "frame_id" in scene:
            payload["frame_id"] = scene.get("frame_id")
        if "timestamp" in scene:
            payload["scene_timestamp"] = scene.get("timestamp")
    client.publish(ACT_CMD_TOPIC, json.dumps(payload))
    print(f"[demo_policy_stub] action={payload.get('action')} frame_id={payload.get('frame_id')}", flush=True)
    state["last_action_ts"] = now
    state["action_index"] += 1


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client = mqtt.Client(client_id="demo_policy_stub", protocol=mqtt.MQTTv311)
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
