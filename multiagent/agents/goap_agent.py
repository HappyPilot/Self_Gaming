#!/usr/bin/env python3
"""GOAP agent translating high-level goals into atomic tasks."""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, Optional

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
GOAL_TOPIC = os.getenv("GOALS_TOPIC", "goals/high_level")
TASK_TOPIC = os.getenv("GOAP_TASK_TOPIC", "goap/tasks")
STATE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
MEM_QUERY_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query")
MEM_REPLY_TOPIC = os.getenv("MEM_REPLY_TOPIC", "mem/reply")


class GOAPAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="goap_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.current_goal: Optional[Dict] = None
        self.last_state: Optional[Dict] = None
        self.pending_queries = {}

    def on_connect(self, client, _userdata, _flags, rc):
        topics = [(GOAL_TOPIC, 0), (STATE_TOPIC, 0), (MEM_REPLY_TOPIC, 0)]
        if rc == 0:
            client.subscribe(topics)
            client.publish(
                TASK_TOPIC,
                json.dumps({"ok": True, "event": "goap_ready"}),
            )
        else:
            client.publish(
                TASK_TOPIC,
                json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}),
            )

    def on_message(self, client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload}

        if msg.topic == GOAL_TOPIC:
            self.current_goal = data
            self.plan_from_goal()
        elif msg.topic == STATE_TOPIC and data.get("ok"):
            self.last_state = data
        elif msg.topic == MEM_REPLY_TOPIC:
            req_id = data.get("request_id")
            if req_id and req_id in self.pending_queries:
                del self.pending_queries[req_id]

    def plan_from_goal(self):
        if not self.current_goal:
            return
        goal_id = self.current_goal.get("goal_id") or uuid.uuid4().hex[:8]
        goal_type = (self.current_goal.get("goal_type") or "explore").lower()
        if goal_type == "farm":
            tasks = [self.make_task(goal_id, "MOVE_TO"), self.make_task(goal_id, "ATTACK_TARGET")]
        elif goal_type == "loot":
            tasks = [self.make_task(goal_id, "MOVE_TO"), self.make_task(goal_id, "LOOT_NEARBY")]
        else:
            tasks = [self.make_task(goal_id, "MOVE_TO"), self.make_task(goal_id, "ATTACK_TARGET"), self.make_task(goal_id, "WAIT")]
        for task in tasks:
            self.client.publish(TASK_TOPIC, json.dumps({"ok": True, **task}))

    def make_task(self, goal_id: str, action_type: str) -> Dict:
        task_id = f"task_{uuid.uuid4().hex[:6]}"
        target = {"x": 0.5, "y": 0.5}
        if self.last_state:
            objects = self.last_state.get("objects") or []
            if action_type == "ATTACK_TARGET":
                enemies = [o for o in objects if "enemy" in str(o.get("class", ""))]
                if enemies:
                    enemy = enemies[0]
                    target = {"x": enemy.get("pos", [0.5, 0.5])[0], "y": enemy.get("pos", [0.5, 0.5])[1], "entity_id": enemy.get("id")}
            elif action_type == "LOOT_NEARBY":
                loot = [o for o in objects if "loot" in str(o.get("class", ""))]
                if loot:
                    item = loot[0]
                    target = {"x": item.get("pos", [0.5, 0.5])[0], "y": item.get("pos", [0.5, 0.5])[1], "entity_id": item.get("id")}
        task = {
            "goal_id": goal_id,
            "task_id": task_id,
            "action_type": action_type,
            "target": target,
            "constraints": {"max_time": 5.0, "max_hp_loss": 0.2},
            "status": "pending",
        }
        return task

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()


def main():
    agent = GOAPAgent()
    agent.run()


if __name__ == "__main__":
    main()
