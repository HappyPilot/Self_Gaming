#!/usr/bin/env python3
import json
import os
import signal
import threading
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EVAL_CMD_TOPIC = os.getenv("EVAL_CMD_TOPIC", "eval/cmd")
EVAL_CMD_ALIAS = os.getenv("EVAL_CMD_ALIAS", "eval/request")
EVAL_RESULT_TOPIC = os.getenv("EVAL_RESULT_TOPIC", "eval/result")
EVAL_RESULT_ALIAS = os.getenv("EVAL_RESULT_ALIAS", "eval/report")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")

stop_event = threading.Event()

def _as_int(code) -> int:
    """Safely convert a paho-mqtt v2 reason code to int."""
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


class EvalAgent:
    """A stub agent for evaluating plans against scene states."""

    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="eval_agent")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.last_scene = {}

    def _publish_eval(self, payload: dict):
        """Publishes a message to all configured result topics."""
        for topic in {EVAL_RESULT_TOPIC, EVAL_RESULT_ALIAS}:
            if topic:
                self.client.publish(topic, json.dumps(payload))

    def on_connect(self, client, userdata, flags, rc):
        """Handles MQTT connection."""
        topics = {(EVAL_CMD_TOPIC, 0), (SCENE_TOPIC, 0)}
        if EVAL_CMD_ALIAS and EVAL_CMD_ALIAS != EVAL_CMD_TOPIC:
            topics.add((EVAL_CMD_ALIAS, 0))

        if _as_int(rc) == 0:
            client.subscribe(list(topics))
            self._publish_eval({"ok": True, "event": "eval_agent_ready"})
        else:
            self._publish_eval({"ok": False, "event": "connect_failed", "code": _as_int(rc)})

    def _run_eval(self, plan_id: str, scene: dict) -> dict:
        """Placeholder evaluation logic."""
        time.sleep(0.5)  # Simulate work
        # Deterministic score based on plan_id hash for mock reproducibility
        score_seed = hash(plan_id) % 10
        score = round(0.5 + 0.5 * (score_seed / 10), 2)
        return {
            "ok": True,
            "plan_id": plan_id,
            "scene_timestamp": scene.get("timestamp"),
            "score": score,
            "timestamp": time.time(),
        }

    def on_message(self, client, userdata, msg):
        """Handles incoming messages for scenes and eval commands."""
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
            if not isinstance(data, dict):
                return
        except json.JSONDecodeError:
            return

        if msg.topic == SCENE_TOPIC and data.get("ok"):
            self.last_scene = data
        elif msg.topic in (EVAL_CMD_TOPIC, EVAL_CMD_ALIAS):
            if not self.last_scene:
                self._publish_eval({
                    "ok": False,
                    "error": "no_scene_available",
                    "note": "Agent has not received a valid scene state yet.",
                })
                return

            plan_id = data.get("plan_id") or f"plan_{uuid4().hex[:6]}"
            result = self._run_eval(plan_id, self.last_scene)
            self._publish_eval(result)

    def start(self):
        """Connects to MQTT and starts the agent's loop."""
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()

def _handle_signal(signum, frame):
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = EvalAgent()
    agent.start()


if __name__ == "__main__":
    main()
