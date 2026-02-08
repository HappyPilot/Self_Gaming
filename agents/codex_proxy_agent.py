#!/usr/bin/env python3
import json
import os
import signal
import threading
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
CODEX_CMD_TOPIC = os.getenv("CODEX_CMD_TOPIC", "codex/cmd")
CODEX_REPLY_TOPIC = os.getenv("CODEX_REPLY_TOPIC", "codex/reply")
MEM_TOPIC = os.getenv("MEM_TOPIC", "mem/store")

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class CodexProxyAgent:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="codex_proxy")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(CODEX_CMD_TOPIC, 0)])
            client.publish(CODEX_REPLY_TOPIC, json.dumps({"ok": True, "event": "codex_proxy_ready"}))
        else:
            client.publish(CODEX_REPLY_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _summarize_request(self, task):
        return {
            "id": task.get("id") or f"cx_{uuid4().hex[:6]}",
            "intent": task.get("intent", "unknown"),
            "files": task.get("files", []),
            "timestamp": time.time(),
        }

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            data = {"raw": msg.payload.decode("utf-8", "ignore")}

        summary = self._summarize_request(data if isinstance(data, dict) else {"raw": data})
        client.publish(MEM_TOPIC, json.dumps({"op": "append", "key": "codex_tasks", "value": summary}))
        client.publish(
            CODEX_REPLY_TOPIC,
            json.dumps({
                "ok": True,
                "event": "codex_task_recorded",
                "summary": summary,
                "note": "Call Codex manually if required",
            }),
        )

    def run(self):
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
    agent = CodexProxyAgent()
    agent.run()

if __name__ == "__main__":
    main()
