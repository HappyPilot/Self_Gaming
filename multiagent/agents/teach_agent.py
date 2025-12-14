#!/usr/bin/env python3
import json
import logging
import os
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

LOG_LEVEL = os.getenv("TEACH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="[teach_agent] %(levelname)s %(message)s")
logger = logging.getLogger("teach_agent")

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

pending_dataset = {}


def _publish_teach(client, payload):
    for topic in {TEACH_CMD_TOPIC, TEACH_CMD_ALIAS}:
        if topic:
            client.publish(topic, json.dumps(payload))
    logger.debug("publish_teach %s", payload)


def on_connect(client, userdata, flags, rc):
    topics = {(SCENE_TOPIC, 0), (TEACH_CMD_TOPIC, 0)}
    if TEACH_CMD_ALIAS and TEACH_CMD_ALIAS != TEACH_CMD_TOPIC:
        topics.add((TEACH_CMD_ALIAS, 0))
    if rc == 0:
        client.subscribe(list(topics))
        _publish_teach(client, {"ok": True, "event": "teach_agent_ready"})
        logger.info("Connected to MQTT; subscribed to %s", topics)
    else:
        _publish_teach(client, {"ok": False, "event": "connect_failed", "code": int(rc)})
        logger.error("Failed to connect rc=%s", rc)


def plan_job(scene_payload, mode="imitation"):
    job_id = f"job_{uuid4().hex[:8]}"
    dataset_id = pending_dataset.get("id") or f"ds_{uuid4().hex[:6]}"
    payload = {
        "ok": True,
        "event": "train_job_created",
        "job_id": job_id,
        "dataset": dataset_id,
        "mode": mode,
        "scene": scene_payload,
        "timestamp": time.time(),
        "teacher": {
            "alpha_start": TEACHER_ALPHA_START,
            "decay_steps": TEACHER_ALPHA_DECAY_STEPS,
        },
        "reward_topic": REWARD_TOPIC,
    }
    return payload


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        data = {"raw": payload}

    if msg.topic == SCENE_TOPIC:
        if data.get("ok") and data.get("event") == "scene_update":
            pending_dataset["last_scene"] = data
            print(f"[teach_agent] cached scene mean={data.get('mean')} text={data.get('text')}", flush=True)
    elif msg.topic in (TEACH_CMD_TOPIC, TEACH_CMD_ALIAS):
        cmd = (data.get("cmd") or "").lower() if isinstance(data, dict) else payload.lower()
        print(f"[teach_agent] command {cmd}", flush=True)
        if cmd in ("plan", "train", "imitation"):
            scene_payload = pending_dataset.get("last_scene")
            if not scene_payload:
                print("[teach_agent] no scene cached; cannot plan", flush=True)
                _publish_teach(client, {"ok": False, "error": "no_scene"})
                return
            job = plan_job(scene_payload, mode="imitation")
            client.publish(TRAIN_JOB_TOPIC, json.dumps(job))
            client.publish(MEM_TOPIC, json.dumps({"op": "append", "key": "train_jobs", "value": job}))
            print(f"[teach_agent] published job {job.get('job_id')}", flush=True)
        elif cmd in ("dataset", "capture"):
            dataset_id = f"ds_{uuid4().hex[:6]}"
            pending_dataset.update({"id": dataset_id, "created": time.time()})
            client.publish(MEM_TOPIC, json.dumps({"op": "set", "key": "current_dataset", "value": pending_dataset}))
            _publish_teach(client, {"ok": True, "event": "dataset_ready", "dataset": dataset_id})
            print(f"[teach_agent] dataset command -> {dataset_id}", flush=True)


def main():
    client = mqtt.Client(client_id="teach_agent", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
