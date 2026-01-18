#!/usr/bin/env python3
"""Automatic training trigger for the behavior-cloning pipeline.

This agent keeps an eye on the recorder dataset directory and MQTT train
statuses. When enough fresh samples accumulate and no job is active, it asks
`teach_agent` to enqueue a new training job by publishing a command on
`teach/request` (and alias).
"""
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Set

import paho.mqtt.client as mqtt

AGENT_NAME = os.getenv("AUTO_TRAIN_AGENT_NAME", "auto_trainer")
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
RECORDER_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
TEACH_TOPIC = os.getenv("TEACH_TOPIC", "teach/request")
TEACH_ALIAS = os.getenv("TEACH_ALIAS", "teach/cmd")
TRAIN_STATUS_TOPIC = os.getenv("TRAIN_STATUS_TOPIC", "train/status")
CHECK_INTERVAL = float(os.getenv("AUTO_TRAIN_CHECK_INTERVAL", "10"))
MIN_SAMPLES = int(os.getenv("AUTO_TRAIN_MIN_SAMPLES", "20"))
MIN_INCREMENT = int(os.getenv("AUTO_TRAIN_MIN_INCREMENT", "5"))
COOLDOWN_SEC = float(os.getenv("AUTO_TRAIN_COOLDOWN", "120"))
SCENE_FRESH_SEC = float(os.getenv("AUTO_TRAIN_SCENE_MAX_AGE", "5"))
STATUS_TOPIC = os.getenv("AUTO_TRAIN_STATUS_TOPIC", "auto_train/status")

stop_event = threading.Event()


def _as_int(code):
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except Exception:
        return 0


class AutoTrainer:
    def __init__(self):
        self.state_lock = threading.Lock()
        self.client = mqtt.Client(client_id=AGENT_NAME, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self.sample_count = 0
        self.last_seen_count = 0
        self.latest_mtime = 0.0
        self.last_seen_mtime = 0.0
        self.last_trigger_time = 0.0
        self.active_job_id: Optional[str] = None
        self.last_scene_ts = 0.0

    # MQTT -----------------------------------------------------------
    def on_connect(self, client, _userdata, _flags, rc, _properties=None):
        if _as_int(rc) == 0:
            subs = [(TRAIN_STATUS_TOPIC, 0), ("scene/state", 0)]
            client.subscribe(subs)
            self._publish_status({"ok": True, "event": "auto_trainer_ready"})
        else:
            self._publish_status({"ok": False, "event": "connect_failed", "code": _as_int(rc)})

    def on_disconnect(self, _client, _userdata, _disconnect_flags, reason_code, _properties=None):
        self._publish_status({"ok": False, "event": "disconnected", "code": _as_int(reason_code)})

    def on_message(self, _client, _userdata, msg):
        with self.state_lock:
            payload = msg.payload.decode("utf-8", "ignore")
            try:
                data = json.loads(payload)
            except Exception:
                data = {}
            if msg.topic == TRAIN_STATUS_TOPIC:
                self._handle_train_status(data)
            elif msg.topic == "scene/state" and isinstance(data, dict) and data.get("event") == "scene_update":
                self.last_scene_ts = time.time()

    def _handle_train_status(self, data: dict):
        job_id = data.get("job_id")
        event = data.get("event")
        if not job_id or not event:
            return
        if event == "job_started":
            self.active_job_id = job_id
            self._publish_status({"ok": True, "event": "job_started", "job_id": job_id})
        elif event in {"job_finished", "job_failed"}:
            self.active_job_id = None
            self.last_seen_count = self.sample_count
            self.last_seen_mtime = self.latest_mtime
            self.last_trigger_time = time.time()
            self._publish_status({"ok": True, "event": event, "job_id": job_id})

    # Core -----------------------------------------------------------
    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()
        try:
            while not stop_event.is_set():
                with self.state_lock:
                    self.sample_count = self._count_samples()
                    self.latest_mtime = self._latest_mtime()
                    if self._should_trigger():
                        self._request_training()
                time.sleep(CHECK_INTERVAL)
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def _count_samples(self) -> int:
        if not RECORDER_DIR.exists():
            return 0
        return sum(1 for _ in RECORDER_DIR.glob("*.json"))

    def _latest_mtime(self) -> float:
        if not RECORDER_DIR.exists():
            return 0.0
        latest = 0.0
        for path in RECORDER_DIR.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
        return latest

    def _should_trigger(self) -> bool:
        if self.sample_count < MIN_SAMPLES:
            return False
        if self.active_job_id is not None:
            return False
        has_new_count = (self.sample_count - self.last_seen_count) >= MIN_INCREMENT
        has_new_mtime = self.latest_mtime > self.last_seen_mtime
        if not (has_new_count or has_new_mtime):
            return False
        now = time.time()
        if now - self.last_trigger_time < COOLDOWN_SEC:
            return False
        if now - self.last_scene_ts > SCENE_FRESH_SEC:
            # Avoid triggering when teach_agent may be missing a recent scene
            return False
        return True

    def _request_training(self):
        payload = {
            "cmd": "plan",
            "source": AGENT_NAME,
            "samples": self.sample_count,
            "timestamp": time.time(),
        }
        for topic in _teach_topics():
            self.client.publish(topic, json.dumps(payload), qos=0)
        self.last_trigger_time = time.time()
        self._publish_status({"ok": True, "event": "plan_requested", "samples": self.sample_count})

    def _publish_status(self, message: dict):
        self.client.publish(STATUS_TOPIC, json.dumps(message), qos=0)


def _teach_topics():
    topics = {TEACH_TOPIC}
    if TEACH_ALIAS and TEACH_ALIAS != TEACH_TOPIC:
        topics.add(TEACH_ALIAS)
    return topics


def _handle_signal(_signum, _frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    trainer = AutoTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
