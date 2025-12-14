#!/usr/bin/env python3
import json
import logging
import os
import signal
import threading
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

# --- Constants ---
LOG_LEVEL = os.getenv("TEACH_LOG_LEVEL", "INFO").upper()
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
TEACH_CMD_TOPIC = os.getenv("TEACH_CMD_TOPIC", "teach/cmd")
TEACH_CMD_ALIAS = os.getenv("TEACH_CMD_ALIAS", "teach/request")
TRAIN_JOB_TOPIC = os.getenv("TRAIN_JOB_TOPIC", "train/jobs")
MEM_TOPIC = os.getenv("MEM_TOPIC", "mem/store")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
TEACHER_ALPHA_START = float(os.getenv("TEACHER_ALPHA_START", "1.0"))
TEACHER_ALPHA_DECAY_STEPS = int(os.getenv("TEACHER_ALPHA_DECAY_STEPS", "500"))

# --- Setup ---
logging.basicConfig(level=LOG_LEVEL, format="[teach_agent] %(levelname)s %(message)s")
logger = logging.getLogger("teach_agent")
stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class TeachAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="teach_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.pending_dataset = {}

    def _publish_teach(self, payload):
        for topic in {TEACH_CMD_TOPIC, TEACH_CMD_ALIAS}:
            if topic:
                self.client.publish(topic, json.dumps(payload))
        logger.debug("publish_teach %s", payload)

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            topics = [(SCENE_TOPIC, 0), (TEACH_CMD_TOPIC, 0)]
            if TEACH_CMD_ALIAS and TEACH_CMD_ALIAS != TEACH_CMD_TOPIC:
                topics.append((TEACH_CMD_ALIAS, 0))
            client.subscribe(topics)
            self._publish_teach({"ok": True, "event": "teach_agent_ready"})
            logger.info("Connected to MQTT; subscribed to %s", [t for t, _ in topics])
        else:
            msg = {"ok": False, "event": "connect_failed", "code": _as_int(rc)}
            logger.error("Failed to connect: rc=%s", _as_int(rc))
            self._publish_teach(msg)

    def _plan_job(self, scene_payload, mode="imitation"):
        job_id = f"job_{uuid4().hex[:8]}"
        dataset_id = self.pending_dataset.get("id") or f"ds_{uuid4().hex[:6]}"
        payload = {
            "ok": True, "event": "train_job_created", "job_id": job_id,
            "dataset": dataset_id, "mode": mode, "scene": scene_payload,
            "timestamp": time.time(),
            "teacher": {"alpha_start": TEACHER_ALPHA_START, "decay_steps": TEACHER_ALPHA_DECAY_STEPS},
            "reward_topic": REWARD_TOPIC,
        }
        return payload

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            data = {"raw": msg.payload}

        if msg.topic == SCENE_TOPIC:
            if isinstance(data, dict) and data.get("ok") and data.get("event") == "scene_update":
                self.pending_dataset["last_scene"] = data
                logger.info("Cached scene mean=%s", data.get('mean'))
        elif msg.topic in (TEACH_CMD_TOPIC, TEACH_CMD_ALIAS):
            cmd = (data.get("cmd") or "").lower() if isinstance(data, dict) else str(data).lower()
            logger.info("Received command: %s", cmd)
            if cmd in ("plan", "train", "imitation"):
                scene_payload = self.pending_dataset.get("last_scene")
                if not scene_payload:
                    logger.warning("No scene cached; cannot plan a job.")
                    self._publish_teach({"ok": False, "error": "no_scene"})
                    return
                job = self._plan_job(scene_payload, mode="imitation")
                client.publish(TRAIN_JOB_TOPIC, json.dumps(job))
                client.publish(MEM_TOPIC, json.dumps({"op": "append", "key": "train_jobs", "value": job}))
                logger.info("Published training job %s", job.get('job_id'))
            elif cmd in ("dataset", "capture"):
                dataset_id = f"ds_{uuid4().hex[:6]}"
                self.pending_dataset.update({"id": dataset_id, "created": time.time()})
                client.publish(MEM_TOPIC, json.dumps({"op": "set", "key": "current_dataset", "value": self.pending_dataset}))
                self._publish_teach({"ok": True, "event": "dataset_ready", "dataset": dataset_id})
                logger.info("Created new dataset ID: %s", dataset_id)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected and shut down.")

def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down.", signum)
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = TeachAgent()
    agent.run()

if __name__ == "__main__":
    main()
