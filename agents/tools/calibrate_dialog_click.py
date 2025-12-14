#!/usr/bin/env python3
"""Isssue a dialog MOVE/CLICK plan and detect success via scene/state."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
GOAP_TOPIC = os.getenv("GOAP_TASK_TOPIC", "goap/tasks")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
DEFAULT_X = float(os.getenv("GOAP_DIALOG_BUTTON_X", "0.52"))
DEFAULT_Y = float(os.getenv("GOAP_DIALOG_BUTTON_Y", "0.80"))
DEFAULT_PROFILE = os.getenv("GOAP_DIALOG_PROFILE", "default")
POLICY_WIDTH = int(os.getenv("POLICY_SCREEN_WIDTH", "1920"))
POLICY_HEIGHT = int(os.getenv("POLICY_SCREEN_HEIGHT", "1080"))
OFFSET_X = int(os.getenv("POLICY_OFFSET_X", "0"))
OFFSET_Y = int(os.getenv("POLICY_OFFSET_Y", "0"))
CAL_TIMEOUT = float(os.getenv("CALIBRATION_TIMEOUT", "20"))


def publish_plan(client: mqtt.Client, x_norm: float, y_norm: float, profile: str) -> None:
    goal_id = f"calibration_{uuid.uuid4().hex[:6]}"
    target = {
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x": x_norm,
        "y": y_norm,
        "area": "critical_dialog",
        "profile": profile,
    }
    move_task = {
        "ok": True,
        "status": "pending",
        "goal_id": goal_id,
        "task_id": f"task_move_{goal_id}",
        "action_type": "MOVE_TO",
        "target": target,
    }
    click_task = dict(move_task)
    click_task.update(
        {
            "task_id": f"task_click_{goal_id}",
            "action_type": "CLICK_BUTTON",
        }
    )
    client.publish(GOAP_TOPIC, json.dumps(move_task))
    time.sleep(0.2)
    client.publish(GOAP_TOPIC, json.dumps(click_task))


def infer_scope(scene: Dict) -> str:
    flags = scene.get("flags") or {}
    if flags.get("death"):
        return "critical_dialog:death"
    return "generic_ui"


def record_success(client: mqtt.Client, profile: str, x_norm: float, y_norm: float, scope: str, scene_text: str):
    if not MEM_STORE_TOPIC:
        return
    payload = {
        "op": "calibration_success",
        "value": {
            "profile": profile,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "timestamp": time.time(),
            "scope": scope,
            "scene_text": scene_text,
        },
    }
    client.publish(MEM_STORE_TOPIC, json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description="Dialog click calibration helper")
    parser.add_argument("--x", type=float, default=DEFAULT_X, help="Normalized X (0-1)")
    parser.add_argument("--y", type=float, default=DEFAULT_Y, help="Normalized Y (0-1)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Dialog profile name")
    parser.add_argument("--timeout", type=float, default=CAL_TIMEOUT, help="Seconds to wait for success")
    args = parser.parse_args()

    state = {
        "observed_death": False,
        "success": False,
        "success_scope": None,
        "scene_text": "",
    }
    success_event = threading.Event()

    client = mqtt.Client(client_id=f"calibrate_{uuid.uuid4().hex[:6]}")

    def on_message(_cli, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {}
        if msg.topic == ACT_CMD_TOPIC:
            if isinstance(data, dict) and data.get("action") in {"mouse_move", "click_primary", "click_secondary"}:
                print(
                    f"act/cmd => {data.get('action')} dx={data.get('dx')} dy={data.get('dy')} "
                    f"target_norm={data.get('target_norm')} target_px={data.get('target_px')}"
                )
        elif msg.topic == SCENE_TOPIC and isinstance(data, dict) and data.get("ok"):
            flags = data.get("flags") or {}
            if flags.get("death"):
                state["observed_death"] = True
            elif state["observed_death"] and not state["success"]:
                state["success"] = True
                state["success_scope"] = infer_scope(data)
                text_entries = data.get("text") or []
                state["scene_text"] = " ".join(text_entries)
                success_event.set()

    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.subscribe(ACT_CMD_TOPIC)
    if SCENE_TOPIC:
        client.subscribe(SCENE_TOPIC)
    client.loop_start()
    try:
        publish_plan(client, args.x, args.y, args.profile)
        print(
            f"Published calibration tasks at x_norm={args.x:.3f}, y_norm={args.y:.3f} "
            f"profile={args.profile} (screen {POLICY_WIDTH}x{POLICY_HEIGHT} offsets {OFFSET_X}/{OFFSET_Y})."
        )
        deadline = time.time() + args.timeout
        while time.time() < deadline and not success_event.is_set():
            time.sleep(0.2)
        if success_event.is_set():
            print(
                f"Calibration success for profile={args.profile} scope={state['success_scope']}"
            )
            record_success(
                client,
                args.profile,
                args.x,
                args.y,
                state.get("success_scope") or "generic_ui",
                state.get("scene_text", ""),
            )
        else:
            print("Calibration did not clear the dialog within timeout.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
