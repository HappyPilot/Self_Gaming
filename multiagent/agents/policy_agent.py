#!/usr/bin/env python3
"""Policy agent that blends heuristic/goap behavior with teacher and PPO weights."""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import paho.mqtt.client as mqtt
import torch
import torch.nn as nn

from models.backbone import Backbone

logging.basicConfig(level=os.getenv("POLICY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("policy_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OBS_TOPIC = os.getenv("OBS_TOPIC", "scene/state")
SIM_TOPIC = os.getenv("SIM_TOPIC", "sim_core/state")
GOAP_TOPIC = os.getenv("GOAP_TASK_TOPIC", "goap/tasks")
CONTROL_TOPIC = os.getenv("ACT_TOPIC", "control/keys")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
TEACHER_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action")
CHECKPOINT_TOPIC = os.getenv("CHECKPOINT_TOPIC", "train/checkpoints")
THRESH = float(os.getenv("THRESH", "120.0"))
DEBOUNCE = float(os.getenv("DEBOUNCE", "0.25"))
TEACHER_ALPHA_START = float(os.getenv("TEACHER_ALPHA_START", "1.0"))
TEACHER_DECAY_STEPS = int(os.getenv("TEACHER_ALPHA_DECAY_STEPS", "500"))
MIN_ALPHA = float(os.getenv("TEACHER_ALPHA_MIN", "0.0"))
MOUSE_RANGE = int(os.getenv("POLICY_MOUSE_RANGE", "60"))
MIN_MOUSE_DELTA = int(os.getenv("POLICY_MIN_MOUSE_DELTA", "8"))
CLICK_COOLDOWN = float(os.getenv("POLICY_CLICK_COOLDOWN", "0.75"))
NON_VISUAL_DIM = 128
NUMERIC_DIM = 32
OBJECT_HIST_DIM = 32
TEXT_EMBED_DIM = 64
FRAME_SHAPE = (3, int(os.getenv("POLICY_FRAME_HEIGHT", "96")), int(os.getenv("POLICY_FRAME_WIDTH", "54")))
BACKBONE_PATH = Path(os.getenv("POLICY_BACKBONE_PATH", "/mnt/ssd/models/backbone/backbone.pt"))
POLICY_HEAD_PATH = Path(os.getenv("POLICY_HEAD_PATH", "/mnt/ssd/models/heads/ppo/policy_head.pt"))
VALUE_HEAD_PATH = Path(os.getenv("POLICY_VALUE_HEAD_PATH", "/mnt/ssd/models/heads/ppo/value_head.pt"))
LABEL_MAP_PATH = Path(os.getenv("POLICY_LABEL_MAP_PATH", "/mnt/ssd/models/heads/ppo/label_map.json"))
HOT_RELOAD_ENABLED = os.getenv("POLICY_HOT_RELOAD", "1") != "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJECT_CLASS_BUCKETS = {
    "enemy_melee": 0,
    "enemy_ranged": 1,
    "boss": 2,
    "projectile": 3,
    "loot_currency": 4,
    "loot_rare": 5,
    "portal": 6,
    "npc": 7,
    "chest": 8,
    "hazard": 9,
}


def _object_bin(label: str) -> int:
    key = label.lower()
    if key in OBJECT_CLASS_BUCKETS:
        return OBJECT_CLASS_BUCKETS[key]
    return hash(key) % OBJECT_HIST_DIM


def encode_non_visual(state: Dict) -> torch.Tensor:
    vector = torch.zeros(NON_VISUAL_DIM, dtype=torch.float32)
    numeric = torch.zeros(NUMERIC_DIM, dtype=torch.float32)
    objects = state.get("objects") or []
    text_entries = state.get("text") or []
    numeric[0] = float(state.get("mean", state.get("mean_brightness", 0.0)))
    numeric[1] = float(len(objects))
    numeric[2] = float(len(text_entries))
    stats = state.get("stats") or {}
    numeric[3] = float(stats.get("hp_pct", 0.0))
    numeric[4] = float(stats.get("enemy_count", 0.0))
    numeric[5] = float(stats.get("loot_count", 0.0))
    vector[:NUMERIC_DIM] = numeric

    hist = torch.zeros(OBJECT_HIST_DIM, dtype=torch.float32)
    for obj in objects:
        label = str(obj.get("class") or obj.get("label") or "unknown")
        hist[_object_bin(label)] += 1.0
    start = NUMERIC_DIM
    vector[start : start + OBJECT_HIST_DIM] = hist

    text_bins = torch.zeros(TEXT_EMBED_DIM, dtype=torch.float32)
    for entry in text_entries:
        for token in str(entry).lower().split():
            idx = hash(token) % TEXT_EMBED_DIM
            text_bins[idx] += 1.0
    vector[start + OBJECT_HIST_DIM :] = text_bins
    return vector


class PolicyAgent:
    """Combines a baseline policy with teacher suggestions via annealing."""

    def __init__(self):
        self.client = mqtt.Client(client_id="policy", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.last_action_ts = 0.0
        self.teacher_action: Optional[dict] = None
        self.latest_state: Optional[dict] = None
        self.current_task: Optional[dict] = None
        self.teacher_alpha_start = max(0.0, TEACHER_ALPHA_START)
        self.teacher_decay_steps = max(1, TEACHER_DECAY_STEPS)
        self.teacher_min_alpha = max(0.0, MIN_ALPHA)
        self.steps = 0
        self.last_label: Optional[str] = None
        self.last_click_ts = 0.0
        self.rng = random.Random()
        self.model_lock = threading.Lock()
        self.model: Optional[Dict] = None
        self.hot_reload_enabled = HOT_RELOAD_ENABLED
        if self.hot_reload_enabled:
            self._initial_model_load()

    # ------------------------------------------------------------------ Model
    def _initial_model_load(self):
        if BACKBONE_PATH.exists() and POLICY_HEAD_PATH.exists() and LABEL_MAP_PATH.exists():
            logger.info("Attempting initial model load from %s", POLICY_HEAD_PATH.parent)
            self._reload_worker(
                {
                    "backbone_path": str(BACKBONE_PATH),
                    "policy_head_path": str(POLICY_HEAD_PATH),
                    "value_head_path": str(VALUE_HEAD_PATH),
                    "label_map_path": str(LABEL_MAP_PATH),
                }
            )
        else:
            logger.warning("Policy hot reload enabled but no initial checkpoint found")

    def _build_model(self, paths: Dict) -> Dict:
        label_map_path = Path(paths.get("label_map_path") or LABEL_MAP_PATH)
        with label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)
        idx_to_action = {int(idx): action for action, idx in label_map.items()}
        num_classes = len(idx_to_action)
        if num_classes == 0:
            raise ValueError("label_map is empty")

        backbone = Backbone(frame_shape=FRAME_SHAPE, non_visual_dim=NON_VISUAL_DIM).to(DEVICE)
        policy_head = nn.Linear(backbone.output_dim, num_classes).to(DEVICE)
        value_head = nn.Linear(backbone.output_dim, 1).to(DEVICE)

        backbone_state = torch.load(paths.get("backbone_path"), map_location=DEVICE)
        backbone.load_state_dict(backbone_state)
        policy_state = torch.load(paths.get("policy_head_path"), map_location=DEVICE)
        policy_head.load_state_dict(policy_state)
        value_state_path = paths.get("value_head_path")
        if value_state_path and Path(value_state_path).exists():
            value_state = torch.load(value_state_path, map_location=DEVICE)
            value_head.load_state_dict(value_state, strict=False)
        backbone.eval()
        policy_head.eval()
        value_head.eval()
        return {
            "backbone": backbone,
            "policy_head": policy_head,
            "value_head": value_head,
            "idx_to_action": idx_to_action,
        }

    def _schedule_model_reload(self, paths: Dict):
        required = ["backbone_path", "policy_head_path", "label_map_path"]
        if not all(paths.get(key) for key in required):
            logger.warning("Checkpoint missing required paths: %s", paths)
            return
        threading.Thread(target=self._reload_worker, args=(paths,), daemon=True).start()

    def _reload_worker(self, paths: Dict):
        try:
            model = self._build_model(paths)
        except Exception as exc:
            logger.error("Failed to load policy checkpoint: %s", exc)
            return
        with self.model_lock:
            self.model = model
        logger.info(
            "Policy model reloaded from %s",
            paths.get("policy_head_path"),
        )

    def _get_model(self) -> Optional[Dict]:
        with self.model_lock:
            return self.model

    # ------------------------------------------------------------------ MQTT
    def _on_connect(self, client, _userdata, _flags, rc):
        subscriptions = [(OBS_TOPIC, 0), (TEACHER_TOPIC, 0)]
        if SIM_TOPIC:
            subscriptions.append((SIM_TOPIC, 0))
        if GOAP_TOPIC:
            subscriptions.append((GOAP_TOPIC, 0))
        if CHECKPOINT_TOPIC:
            subscriptions.append((CHECKPOINT_TOPIC, 0))
        if rc == 0:
            client.subscribe(subscriptions)
            logger.info(
                "Policy agent connected; subscribed to %s",
                ", ".join(topic for topic, _ in subscriptions),
            )
        else:
            logger.error("Policy agent failed to connect: rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        logger.warning("Policy agent disconnected rc=%s", rc)

    def _on_message(self, client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {}

        if msg.topic == TEACHER_TOPIC:
            if isinstance(data, dict):
                action_text = data.get("action") or data.get("text")
                if action_text:
                    self.teacher_action = {
                        "text": action_text,
                        "reasoning": data.get("reasoning"),
                        "timestamp": data.get("timestamp", time.time()),
                    }
                    logger.info("Received teacher action: %s", action_text)
        elif msg.topic == OBS_TOPIC or (SIM_TOPIC and msg.topic == SIM_TOPIC):
            self.latest_state = data
            policy_action = self._policy_from_observation(data)
            if policy_action is None:
                return
            final_action = self._blend_with_teacher(policy_action)
            if final_action is None:
                return
            self._publish_action(client, final_action, policy_action)
        elif GOAP_TOPIC and msg.topic == GOAP_TOPIC:
            if data.get("status") == "pending":
                self.current_task = data
        elif CHECKPOINT_TOPIC and msg.topic == CHECKPOINT_TOPIC and self.hot_reload_enabled:
            if isinstance(data, dict):
                self._schedule_model_reload(data)

    # ----------------------------------------------------------------- Policy
    def _policy_from_observation(self, data: dict) -> Optional[Dict[str, object]]:
        state = data or self.latest_state or {}
        try:
            mean_val = float(state.get("mean_brightness", state.get("mean", 0.0)))
        except (TypeError, ValueError):
            mean_val = 0.0
        now = time.time()
        if (now - self.last_action_ts) <= DEBOUNCE:
            return None

        if self.current_task:
            action = self._action_from_task(state)
        else:
            action = self._action_from_model(state)
            if action is None:
                if self.last_label == "mouse_move":
                    action = {"label": "click_primary", "confidence": min(1.0, mean_val / THRESH)}
                else:
                    dx = self._random_delta()
                    dy = self._random_delta()
                    action = {"label": "mouse_move", "dx": dx, "dy": dy, "confidence": min(1.0, mean_val / THRESH)}
        self.last_action_ts = now
        return action

    def _current_alpha(self) -> float:
        progress = min(self.steps / float(self.teacher_decay_steps), 1.0)
        alpha = self.teacher_alpha_start * (1.0 - progress)
        return max(self.teacher_min_alpha, min(1.0, alpha))

    def _blend_with_teacher(self, policy_action: Dict[str, object]) -> Optional[Dict[str, object]]:
        teacher_alpha = self._current_alpha()
        teacher_action = self._teacher_to_action()

        chosen = None
        if teacher_action and teacher_alpha > 0:
            if teacher_alpha >= 0.99:
                chosen = teacher_action
            elif policy_action is None:
                chosen = teacher_action
            else:
                prob = max(0.0, min(1.0, teacher_alpha))
                chosen = teacher_action if self.rng.random() < prob else policy_action
        else:
            chosen = policy_action

        if chosen is None:
            return None

        if chosen.get("label") == "click_primary" and not self._click_ready():
            fallback = policy_action if policy_action and policy_action.get("label") == "mouse_move" else self._fallback_move()
            chosen = fallback

        self.steps += 1
        logger.debug(
            "Policy decision | teacher_alpha=%.3f | teacher=%s | policy=%s | chosen=%s",
            teacher_alpha,
            teacher_action.get("label") if teacher_action else None,
            policy_action.get("label") if policy_action else None,
            chosen.get("label") if chosen else None,
        )
        return chosen

    def _teacher_to_action(self) -> Optional[Dict[str, object]]:
        if not self.teacher_action:
            return None
        text = (self.teacher_action.get("text") or "").lower()
        if not text:
            return None

        if any(word in text for word in ("click", "attack", "shoot", "select")):
            button = "click_secondary" if "right" in text else "click_primary"
            return {"label": button}

        if "hold" in text and "mouse" in text:
            return {"label": "mouse_hold", "button": "left" if "left" in text else "right"}

        if "release" in text and "mouse" in text:
            return {"label": "mouse_release", "button": "left" if "left" in text else "right"}

        move_vec = self._direction_from_text(text)
        if move_vec:
            dx, dy = move_vec
            return {"label": "mouse_move", "dx": dx, "dy": dy}

        key = self._extract_key(text)
        if key:
            return {"label": "key_press", "key": key}

        return None

    def _direction_from_text(self, text: str) -> Optional[tuple]:
        if "move" not in text:
            return None
        dx = dy = 0
        magnitude = max(MIN_MOUSE_DELTA, MOUSE_RANGE // 2)
        if "left" in text:
            dx -= magnitude
        if "right" in text:
            dx += magnitude
        if "up" in text or "north" in text:
            dy -= magnitude
        if "down" in text or "south" in text:
            dy += magnitude
        if dx == 0 and dy == 0:
            dx = self._random_delta()
            dy = self._random_delta()
        return dx, dy

    def _extract_key(self, text: str) -> Optional[str]:
        if "press" not in text:
            return None
        tokens = text.split()
        for token in tokens:
            token = token.strip(".,")
            if len(token) == 1 and token.isalnum():
                return token.lower()
            if token in {"enter", "space", "escape", "tab"}:
                return token
        return None

    def _action_from_task(self, state: dict) -> Dict[str, object]:
        task = self.current_task or {}
        action_type = (task.get("action_type") or "MOVE_TO").upper()
        target = task.get("target") or {}
        dx = target.get("x", 0.5) - 0.5
        dy = target.get("y", 0.5) - 0.5
        scale = MOUSE_RANGE
        if action_type == "MOVE_TO":
            return {
                "label": "mouse_move",
                "dx": int(dx * scale),
                "dy": int(dy * scale),
                "task_id": task.get("task_id"),
            }
        if action_type == "ATTACK_TARGET":
            return {"label": "click_primary", "task_id": task.get("task_id"), "confidence": 1.0}
        if action_type == "LOOT_NEARBY":
            return {"label": "click_secondary", "task_id": task.get("task_id"), "confidence": 1.0}
        return {
            "label": "mouse_move",
            "dx": self._random_delta(),
            "dy": self._random_delta(),
            "task_id": task.get("task_id"),
        }

    def _action_from_model(self, state: dict) -> Optional[Dict[str, object]]:
        model = self._get_model()
        if not model:
            return None
        try:
            non_visual = encode_non_visual(state).unsqueeze(0).to(DEVICE)
            frame = torch.zeros(1, *FRAME_SHAPE, device=DEVICE)
            with torch.no_grad():
                final_state = model["backbone"](frame, non_visual)
                logits = model["policy_head"](final_state)
                idx = torch.argmax(logits, dim=1).item()
            action_label = model["idx_to_action"].get(idx)
            if not action_label:
                return None
            return self._dict_from_label(action_label)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            return None

    def _dict_from_label(self, label: str) -> Dict[str, object]:
        label = str(label).lower()
        if label == "mouse_move":
            return {
                "label": "mouse_move",
                "dx": self._random_delta(),
                "dy": self._random_delta(),
            }
        if label in {"click", "click_primary", "attack"}:
            return {"label": "click_primary", "confidence": 1.0}
        if label in {"click_secondary", "loot"}:
            return {"label": "click_secondary", "confidence": 1.0}
        if label.startswith("key_"):
            key = label.split("_", 1)[1]
            return {"label": "key_press", "key": key}
        if label in {"press_space", "space"}:
            return {"label": "key_press", "key": "space"}
        return {"label": label}

    def _random_delta(self) -> int:
        delta = self.rng.randint(-MOUSE_RANGE, MOUSE_RANGE)
        if delta >= 0:
            delta = max(MIN_MOUSE_DELTA, delta)
        else:
            delta = min(-MIN_MOUSE_DELTA, delta)
        return delta

    def _fallback_move(self) -> Dict[str, object]:
        return {"label": "mouse_move", "dx": self._random_delta(), "dy": self._random_delta()}

    def _click_ready(self) -> bool:
        return (time.time() - self.last_click_ts) >= CLICK_COOLDOWN

    def _key_payload(self, action: Dict[str, object]) -> Optional[dict]:
        label = action.get("label")
        if label == "key_press":
            key = action.get("key")
            return {"key": key} if key else None
        text = str(label or "").lower()
        if "enter" in text:
            return {"key": "enter"}
        if "space" in text:
            return {"key": "space"}
        if "escape" in text or "esc" in text:
            return {"key": "escape"}
        if "tab" in text:
            return {"key": "tab"}
        return None

    # ----------------------------------------------------------------- Publish
    def _publish_action(self, client, chosen: Dict[str, object], policy_action: Optional[Dict[str, object]]):
        label = chosen.get("label")
        key_payload = self._key_payload(chosen)
        act_payload = {"action": label}
        if label == "mouse_move":
            act_payload["dx"] = int(chosen.get("dx", 0))
            act_payload["dy"] = int(chosen.get("dy", 0))
        elif label in {"mouse_hold", "mouse_release"}:
            act_payload["button"] = chosen.get("button", "left")
        elif label == "key_press" and key_payload:
            act_payload["key"] = key_payload.get("key")

        envelope = {
            "ok": True,
            "source": "policy_agent",
            "action": label,
            "policy_action": policy_action.get("label") if policy_action else None,
            "teacher_action": self.teacher_action.get("text") if self.teacher_action else None,
            "teacher_alpha": round(self._current_alpha(), 3),
            "timestamp": time.time(),
        }
        if self.current_task:
            envelope["task_id"] = self.current_task.get("task_id")
            envelope["goal_id"] = self.current_task.get("goal_id")
        client.publish(ACT_CMD_TOPIC, json.dumps({**envelope, **act_payload}))
        if key_payload:
            client.publish(CONTROL_TOPIC, json.dumps(key_payload))
        logger.info("Published blended action: %s", label)
        if label and str(label).startswith("click"):
            self.last_click_ts = time.time()
        self.last_label = label

    # ----------------------------------------------------------------- Public
    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 60)
        self.client.loop_forever()


def main():
    agent = PolicyAgent()
    agent.start()


if __name__ == "__main__":
    main()
