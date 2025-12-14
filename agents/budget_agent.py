#!/usr/bin/env python3
import json
import os
import time
import logging
from pathlib import Path

import paho.mqtt.client as mqtt

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
BUDGET_TOPIC = os.getenv("BUDGET_TOPIC", "budget/update")
BUDGET_SUMMARY_TOPIC = os.getenv("BUDGET_SUMMARY_TOPIC", "budget/summary")
TOKEN_LIMIT = int(os.getenv("BUDGET_TOKEN_LIMIT", "20000"))
STATE_FILE = os.getenv("BUDGET_STATE_FILE", "budget_state.json")

logging.basicConfig(level=os.getenv("BUDGET_LOG_LEVEL", "INFO"))
logger = logging.getLogger("budget_agent")


def _as_int(code) -> int:
    """Safely convert a paho-mqtt v2 reason code to int."""
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


class BudgetAgent:
    """Tracks token and GPU usage and reports summaries."""

    def __init__(self):
        self.client = mqtt.Client(client_id="budget_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.state_file = Path(STATE_FILE)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Loads the budget state from a file."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    loaded_state = json.load(f)
                    # Basic validation
                    if "tokens" in loaded_state and "gpu_hours" in loaded_state:
                        logger.info("Loaded previous budget state from %s", self.state_file)
                        return loaded_state
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load state from %s: %s", self.state_file, e)
        return {"tokens": 0, "gpu_hours": 0.0}

    def _save_state(self):
        """Saves the current budget state to a file."""
        try:
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            logger.error("Failed to save state to %s: %s", self.state_file, e)

    def on_connect(self, client, userdata, flags, rc):
        rc_int = _as_int(rc)
        if rc_int == 0:
            client.subscribe([(BUDGET_TOPIC, 0)])
            logger.info("Budget agent connected, subscribed to %s", BUDGET_TOPIC)
            self._publish_summary(event="budget_agent_ready")
        else:
            logger.error("Budget agent failed to connect: rc=%s", rc_int)
            self.client.publish(
                BUDGET_SUMMARY_TOPIC,
                json.dumps({"ok": False, "event": "connect_failed", "code": rc_int}),
            )

    def on_message(self, client, userdata, msg):
        """Handles incoming budget update messages."""
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
            if not isinstance(data, dict):
                raise ValueError("Payload is not a dictionary")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Could not parse budget update payload: %s. Payload: %s", e, msg.payload[:100])
            return

        try:
            tokens_to_add = int(data.get("tokens", 0))
            gpu_hours_to_add = float(data.get("gpu_hours", 0.0))
        except (TypeError, ValueError) as e:
            logger.warning("Invalid data types in budget update: %s. Data: %s", e, data)
            return

        self.state["tokens"] += tokens_to_add
        self.state["gpu_hours"] += gpu_hours_to_add

        self._save_state()
        self._publish_summary(event="budget_update")

    def _publish_summary(self, event: str = "budget_summary"):
        """Publishes the current budget summary."""
        summary = {
            "ok": True,
            "event": event,
            "tokens": self.state["tokens"],
            "gpu_hours": round(self.state["gpu_hours"], 3),
            "limit": TOKEN_LIMIT,
            "timestamp": time.time(),
            "throttle": self.state["tokens"] >= TOKEN_LIMIT,
        }
        self.client.publish(BUDGET_SUMMARY_TOPIC, json.dumps(summary))

    def start(self):
        """Connects to MQTT and starts the main loop."""
        try:
            self.client.connect(MQTT_HOST, MQTT_PORT, 30)
            self.client.loop_forever()
        except Exception as e:
            logger.critical("Budget agent failed to start: %s", e)


def main():
    agent = BudgetAgent()
    agent.start()


if __name__ == "__main__":
    main()
