#!/usr/bin/env python3
"""Thermal monitor that watches tegrastats and publishes system/thermal."""
from __future__ import annotations

import json
import os
import subprocess
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
THERMAL_TOPIC = os.getenv("THERMAL_TOPIC", "system/thermal")
INTERVAL = float(os.getenv("THERMAL_INTERVAL", "5"))
COOLDOWN_DURATION = float(os.getenv("THERMAL_COOLDOWN", "900"))
ACTIVE_THRESHOLD = float(os.getenv("THERMAL_ACTIVE_LOAD", "50"))

SOFT_LIMIT = 65.0
HARD_LIMIT = 70.0
ACTIVE_LIMIT = 3600.0


class ThermalMonitor:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="thermal_monitor", protocol=mqtt.MQTTv311)
        self.active_time = 0.0
        self.cooldown_until = 0.0

    def read_stats(self):
        try:
            result = subprocess.run(
                ["tegrastats", "--interval", "1000", "--count", "1"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def parse_temp_usage(self, stats: str):
        temp = 0.0
        usage = 0.0
        for token in stats.replace(',', ' ').split():
            if "GPU@" in token:
                try:
                    temp = float(token.split("@")[1].replace("C", ""))
                except ValueError:
                    continue
            if token.startswith("GR3D_FREQ"):
                parts = token.split()
                try:
                    usage = float(parts[1].replace("%", ""))
                except (IndexError, ValueError):
                    continue
        return temp, usage

    def state_from_temp(self, temp: float):
        if temp >= HARD_LIMIT:
            return "hot"
        if temp >= SOFT_LIMIT:
            return "warm"
        return "ok"

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        try:
            while True:
                stats = self.read_stats()
                temp, usage = self.parse_temp_usage(stats)
                state = self.state_from_temp(temp)
                if usage >= ACTIVE_THRESHOLD and state != "hot":
                    self.active_time += INTERVAL
                else:
                    self.active_time = max(0.0, self.active_time - INTERVAL)

                mode = "normal"
                now = time.time()
                if state == "hot" or self.active_time >= ACTIVE_LIMIT:
                    self.cooldown_until = now + COOLDOWN_DURATION
                if now < self.cooldown_until:
                    mode = "cooldown"

                payload = {
                    "ok": True,
                    "event": "thermal_update",
                    "temp": temp,
                    "gpu_usage": usage,
                    "state": state,
                    "active_minutes": round(self.active_time / 60.0, 2),
                    "mode": mode,
                    "timestamp": now,
                }
                self.client.publish(THERMAL_TOPIC, json.dumps(payload))
                time.sleep(INTERVAL)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def main():
    monitor = ThermalMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
