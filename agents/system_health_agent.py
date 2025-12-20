#!/usr/bin/env python3
"""System Health Agent: Monitors agent activity and signals alarms on failure."""
import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict

import paho.mqtt.client as mqtt

# --- Setup ---
logging.basicConfig(
    level=os.getenv("HEALTH_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("system_health")
stop_event = threading.Event()

# --- Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
HEALTH_TOPIC = os.getenv("HEALTH_TOPIC", "system/health")
ALERT_TOPIC = os.getenv("ALERT_TOPIC", "system/alert")
PAUSE_TOPIC = os.getenv("PAUSE_TOPIC", "act/cmd")

# Thresholds (seconds without message before declaring unhealthy)
THRESHOLDS = {
    "vision": float(os.getenv("TIMEOUT_VISION", "5.0")),  # Camera should stream constantly
    "scene": float(os.getenv("TIMEOUT_SCENE", "10.0")),   # Scene might take time but usually fast
    "policy": float(os.getenv("TIMEOUT_POLICY", "60.0")), # Policy might wait for teacher
}

# Topics to map to services
TOPIC_MAP = {
    "vision/mean": "vision",
    "vision/frame/preview": "vision",
    "vision/frame/full": "vision",
    "vision/frame": "vision",
    "scene/state": "scene",
    "act/cmd": "policy",
    "train/status": "train_manager",
    "world_model/pred_error": "world_model"
}

class SystemHealthAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="system_health", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.last_seen = defaultdict(float)
        self.status = "startup"
        self.lock = threading.Lock()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT. Watching system pulse.")
            # Subscribe to all topics relevant for health monitoring
            topics = [(t, 0) for t in TOPIC_MAP.keys()]
            client.subscribe(topics)
        else:
            logger.error("Failed to connect to MQTT: rc=%s", rc)

    def _on_message(self, client, userdata, msg):
        service = TOPIC_MAP.get(msg.topic)
        if service:
            with self.lock:
                self.last_seen[service] = time.time()

    def _check_health(self):
        now = time.time()
        report = {}
        overall_healthy = True
        
        with self.lock:
            for service, timeout in THRESHOLDS.items():
                last = self.last_seen.get(service, 0.0)
                # Special case: if never seen, allow grace period at startup
                if last == 0.0:
                    age = 999.0 # Treated as unseen
                else:
                    age = now - last
                
                is_alive = age < timeout
                report[service] = {
                    "status": "ok" if is_alive else "critical",
                    "age_sec": round(age, 1)
                }
                
                if not is_alive:
                    # Ignore missing signals during first 30s of uptime
                    if time.time() - self.start_time > 30.0:
                        overall_healthy = False
                        logger.warning("CRITICAL: Service '%s' silent for %.1fs", service, age)

        payload = {
            "ok": overall_healthy,
            "services": report,
            "timestamp": now
        }

        self.client.publish(HEALTH_TOPIC, json.dumps(payload))
        
        if not overall_healthy:
            # publish alert
            self.client.publish(ALERT_TOPIC, json.dumps({
                "event": "system_unhealthy", 
                "details": {k: v for k, v in report.items() if v["status"] != "ok"}
            }))
            # auto-pause actions to avoid runaway when vision lost
            if PAUSE_TOPIC:
                self.client.publish(PAUSE_TOPIC, json.dumps({
                    "action": "wait",
                    "source": "system_health",
                    "reason": "vision_or_scene_unhealthy",
                    "timestamp": now
                }))

    def run(self):
        self.start_time = time.time()
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        
        try:
            while not stop_event.is_set():
                self._check_health()
                stop_event.wait(2.0) # Check every 2 seconds
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Health monitor stopped.")

def _handle_signal(signum, frame):
    logger.info("Signal received, stopping.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = SystemHealthAgent()
    agent.run()

if __name__ == "__main__":
    main()
