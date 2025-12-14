#!/usr/bin/env python3
"""GOAP agent translating high-level goals into atomic tasks."""
from __future__ import annotations

import json
import os
import random
import signal
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

import paho.mqtt.client as mqtt
import logging

try:
    from .mem_rpc import MemRPC  # type: ignore
except ImportError:  # pragma: no cover
    from mem_rpc import MemRPC

logging.basicConfig(level=os.getenv("GOAP_LOG_LEVEL", "INFO"))
logger = logging.getLogger("goap_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
GOAL_TOPIC = os.getenv("GOALS_TOPIC", "goals/high_level")
TASK_TOPIC = os.getenv("GOAP_TASK_TOPIC", "goap/tasks")
STATE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
MEM_QUERY_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query")
MEM_REPLY_TOPIC = os.getenv("MEM_REPLY_TOPIC", "mem/reply")
MEM_RESPONSE_TOPIC = os.getenv("MEM_RESPONSE_TOPIC", "mem/response")
LEARNING_STAGE = int(os.getenv("LEARNING_STAGE", "1"))
DEATH_BUTTON_TEXT = os.getenv("GOAP_DEATH_BUTTON", "resurrect")
DEATH_KEYWORDS = [kw.strip().lower() for kw in os.getenv("GOAP_DEATH_KEYWORDS", "you have died,resurrect,revive,respawn,checkpoint").split(",") if kw.strip()]
DIALOG_PROFILE = os.getenv("GOAP_DIALOG_PROFILE", "poe_default")
DIALOG_SCOPE = os.getenv("GOAP_DIALOG_SCOPE", "critical_dialog:death")
DIALOG_BUTTON_X = float(os.getenv("GOAP_DIALOG_BUTTON_X", os.getenv("GOAP_DEATH_BUTTON_X", "0.5")))
DIALOG_BUTTON_Y = float(os.getenv("GOAP_DIALOG_BUTTON_Y", os.getenv("GOAP_DEATH_BUTTON_Y", "0.82")))
DEATH_HP_THRESHOLD = float(os.getenv("GOAP_DEATH_HP_THRESHOLD", "0.05"))
DEATH_MEAN_THRESHOLD = float(os.getenv("GOAP_DEATH_MEAN_THRESHOLD", "0.22"))
DEATH_SYMBOLS = [sym.strip() for sym in os.getenv("GOAP_DEATH_SYMBOLS", "*,†,+,☠").split(",") if sym.strip()]
RECENT_CRITICAL_LIMIT = int(os.getenv("GOAP_RECENT_CRITICAL_LIMIT", "5"))
CALIBRATION_QUERY_LIMIT = int(os.getenv("GOAP_CALIBRATION_QUERY_LIMIT", "5"))
CALIBRATION_MAX_AGE = float(os.getenv("GOAP_CALIBRATION_MAX_AGE", "7200"))
STUCK_EPISODES_LIMIT = int(os.getenv("GOAP_STUCK_EPISODES_LIMIT", "10"))
STUCK_PLAN_LIMIT = int(os.getenv("GOAP_STUCK_PLAN_LIMIT", "5"))
JITTER_RANGE = float(os.getenv("GOAP_DIALOG_JITTER_RANGE", "0.02"))
DIALOG_TEXT_HINTS = [hint.strip().lower() for hint in os.getenv("GOAP_DIALOG_HINTS", "resurrect,checkpoint").split(",") if hint.strip()]
DIALOG_HINT_MIN_HITS = int(os.getenv("GOAP_DIALOG_HINT_MIN", "1"))

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class GOAPAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(client_id="goap_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.current_goal: Optional[Dict] = None
        self.last_state: Optional[Dict] = None
        self.pending_queries = {}
        self.learning_stage = LEARNING_STAGE
        self.stage0_death_active = False
        self.death_keywords = set(DEATH_KEYWORDS)
        self.mem_rpc: Optional[MemRPC] = None
        self.dialog_plans_since_recover = 0
        self.rng = random.Random()
        self.dialog_text_hints = list(DIALOG_TEXT_HINTS)
        self.dialog_hint_min_hits = max(1, DIALOG_HINT_MIN_HITS)

    def on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            topics = [(GOAL_TOPIC, 0), (STATE_TOPIC, 0), (MEM_REPLY_TOPIC, 0)]
            client.subscribe(topics)
            client.publish(TASK_TOPIC, json.dumps({"ok": True, "event": "goap_ready"}))
        else:
            client.publish(TASK_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    # ... (rest of methods unchanged until run) ...

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
            if self.learning_stage == 0:
                self._handle_stage0_state(data)
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
        elif goal_type in {"recover_from_death", "respawn"}:
            tasks = self._death_tasks(goal_id)
        elif self.learning_stage == 0:
            tasks = [self.make_task(goal_id, "MOVE_TO", target={"x": 0.5, "y": 0.5})]
        else:
            tasks = [self.make_task(goal_id, "MOVE_TO"), self.make_task(goal_id, "ATTACK_TARGET"), self.make_task(goal_id, "WAIT")]
        if not tasks:
            logger.warning(
                "GOAP: goal %s produced no tasks (scope=%s)",
                goal_type,
                (self.current_goal or {}).get("scope", "unknown"),
            )
            return
        self._publish_tasks(tasks)

    def make_task(self, goal_id: str, action_type: str, target: Optional[Dict] = None) -> Dict:
        task_id = f"task_{uuid.uuid4().hex[:6]}"
        resolved_target = {"x": 0.5, "y": 0.5, "x_norm": 0.5, "y_norm": 0.5}
        if self.last_state:
            objects = self.last_state.get("objects") or []
            if action_type == "ATTACK_TARGET":
                enemies = [o for o in objects if "enemy" in str(o.get("class", ""))]
                if enemies:
                    enemy = enemies[0]
                    ex, ey = enemy.get("pos", [0.5, 0.5])
                    resolved_target = {"x_norm": ex, "y_norm": ey, "entity_id": enemy.get("id")}
            elif action_type == "LOOT_NEARBY":
                loot = [o for o in objects if "loot" in str(o.get("class", ""))]
                if loot:
                    item = loot[0]
                    lx, ly = item.get("pos", [0.5, 0.5])
                    resolved_target = {"x_norm": lx, "y_norm": ly, "entity_id": item.get("id")}
        if target:
            resolved_target.update(target)
        task = {
            "goal_id": goal_id,
            "task_id": task_id,
            "action_type": action_type,
            "target": resolved_target,
            "constraints": {"max_time": 5.0, "max_hp_loss": 0.2},
            "status": "pending",
        }
        return task

    def _publish_tasks(self, tasks):
        for task in tasks:
            logger.info(
                "Publishing GOAP task %s (%s) target=%s",
                task.get("task_id"),
                task.get("action_type"),
                task.get("target"),
            )
            self.client.publish(TASK_TOPIC, json.dumps({"ok": True, **task}))

    def _handle_stage0_state(self, state: Dict):
        death, reason = self._detect_death_state(state)
        if death and not self.stage0_death_active:
            self.stage0_death_active = True
            goal_id = f"death_{int(time.time())}"
            tasks = self._death_tasks(goal_id)
            if not tasks:
                logger.warning("Stage0 death detected but dialog missing; waiting for UI")
                return
            logger.info(
                "Stage0 death detected (%s) -> plan %s scope=%s",
                reason,
                [t.get("action_type") for t in tasks],
                DIALOG_SCOPE,
            )
            self._publish_tasks(tasks)
            self._log_recent_failures(DIALOG_SCOPE)
        elif not death and self.stage0_death_active:
            self.stage0_death_active = False
            self.dialog_plans_since_recover = 0

    def _death_tasks(self, goal_id: str):
        """Return MOVE+CLICK tasks aimed at the configured critical dialog button."""

        if not self._death_dialog_visible():
            logger.warning(
                "GOAP: death/recovery goal requested but dialog hints missing (scope=%s)",
                DIALOG_SCOPE,
            )
            return []
        x_norm, y_norm, source = self._resolve_dialog_target()
        stuck_reason, deaths, recoveries = self._evaluate_stuck_dialog()
        jittered = False
        if stuck_reason:
            prev_x, prev_y = x_norm, y_norm
            new_x, new_y = self._apply_jitter(x_norm, y_norm)
            jittered = (new_x, new_y) != (prev_x, prev_y)
            x_norm, y_norm = new_x, new_y
            logger.warning(
                "GOAP: stuck in %s (reason=%s deaths=%s recoveries=%s) -> target (%.3f, %.3f) jittered to (%.3f, %.3f)",
                DIALOG_SCOPE,
                stuck_reason,
                deaths,
                recoveries,
                prev_x,
                prev_y,
                x_norm,
                y_norm,
            )
        target = {
            "x_norm": x_norm,
            "y_norm": y_norm,
            "x": x_norm,
            "y": y_norm,
            "text": DEATH_BUTTON_TEXT,
            "button": "primary",
            "area": "critical_dialog",
            "scope": DIALOG_SCOPE,
            "profile": DIALOG_PROFILE,
        }
        target["target_norm"] = [x_norm, y_norm]
        source_label = source
        if jittered:
            source_label = f"jitter({source})"
        target["target_source"] = source_label
        self.dialog_plans_since_recover += 1
        logger.info(
            "Planning critical dialog recovery profile=%s scope=%s target=(%.3f, %.3f) source=%s",
            DIALOG_PROFILE,
            DIALOG_SCOPE,
            x_norm,
            y_norm,
            source_label,
        )
        return [
            self.make_task(goal_id, "MOVE_TO", target=target),
            self.make_task(goal_id, "CLICK_BUTTON", target=target),
        ]

    def _death_dialog_visible(self) -> bool:
        if not isinstance(self.last_state, dict):
            return False
        flags = self.last_state.get("flags") or {}
        if flags.get("death"):
            return True
        texts = self.last_state.get("text") or []
        if not texts:
            return False
        for entry in texts:
            lowered = str(entry).lower()
            if not lowered:
                continue
            if any(keyword in lowered for keyword in self.death_keywords):
                return True
            hits = sum(1 for hint in self.dialog_text_hints if hint and hint in lowered)
            if hits >= self.dialog_hint_min_hits:
                return True
        return False

    def _detect_death_state(self, state: Dict) -> tuple[bool, str]:
        if not isinstance(state, dict):
            return False, ""
        flags = state.get("flags") or {}
        if flags.get("death"):
            return True, str(state.get("death_reason") or "scene_flag")
        if state.get("death_reason"):
            return True, state["death_reason"]
        text_entries = state.get("text") or []
        lower_text = " ".join(text_entries).lower()
        if lower_text and any(kw in lower_text for kw in self.death_keywords):
            return True, "text_match"
        stats = state.get("stats") or {}
        hp_pct = stats.get("hp_pct")
        if hp_pct is not None:
            try:
                if float(hp_pct) <= DEATH_HP_THRESHOLD:
                    return True, "hp_low"
            except (TypeError, ValueError):
                pass
        objects = state.get("objects") or []
        for obj in objects:
            label = str(obj.get("label") or obj.get("text") or obj.get("class") or "").lower()
            if label and any(kw in label for kw in self.death_keywords):
                return True, "object_match"
        mean_val = state.get("mean")
        if mean_val is not None and text_entries and self._symbolic_text_only(text_entries):
            try:
                if float(mean_val) <= DEATH_MEAN_THRESHOLD:
                    return True, "symbolic_text"
            except (TypeError, ValueError):
                pass
        return False, ""

    def _symbolic_text_only(self, entries):
        if not DEATH_SYMBOLS:
            return False
        for entry in entries:
            cleaned = str(entry).strip()
            if not cleaned:
                continue
            if any(ch.isalpha() for ch in cleaned) or any(ch.isdigit() for ch in cleaned):
                return False
            if not any(sym in cleaned for sym in DEATH_SYMBOLS):
                return False
        return True

    # -------------------------------------------------------------- Memory IO
    def _ensure_mem_rpc(self) -> Optional[MemRPC]:
        if self.mem_rpc:
            return self.mem_rpc
        if not (MEM_QUERY_TOPIC and MEM_RESPONSE_TOPIC):
            return None
        try:
            self.mem_rpc = MemRPC(
                host=MQTT_HOST,
                port=MQTT_PORT,
                query_topic=MEM_QUERY_TOPIC,
                reply_topic=MEM_RESPONSE_TOPIC,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("GOAP could not init mem RPC: %s", exc)
            self.mem_rpc = None
        return self.mem_rpc

    def _log_recent_failures(self, scope: str):
        rpc = self._ensure_mem_rpc()
        if not rpc:
            return
        payload = {"mode": "recent_critical", "scope": scope, "limit": RECENT_CRITICAL_LIMIT}
        response = rpc.query(payload, timeout=0.5)
        if not response:
            return
        entries = response.get("value") or []
        if not entries:
            return
        deaths = sum(1 for entry in entries if entry.get("delta") == "hero_dead")
        recoveries = sum(1 for entry in entries if entry.get("delta") == "hero_resurrected")
        if deaths and deaths > recoveries:
            logger.warning(
                "Recent critical history shows %s deaths vs %s recoveries (scope=%s)",
                deaths,
                recoveries,
                scope,
            )

    def _resolve_dialog_target(self) -> Tuple[float, float, str]:
        """Prefer calibrated targets when available, otherwise fall back to env defaults."""

        x_norm = DIALOG_BUTTON_X
        y_norm = DIALOG_BUTTON_Y
        source = "default_env"
        rpc = self._ensure_mem_rpc()
        if not rpc or CALIBRATION_QUERY_LIMIT <= 0:
            return x_norm, y_norm, source
        payload = {
            "mode": "calibration_events",
            "scope": DIALOG_SCOPE,
            "profile": DIALOG_PROFILE,
            "limit": CALIBRATION_QUERY_LIMIT,
        }
        response = rpc.query(payload, timeout=0.7)
        events = (response or {}).get("value") or []
        now = time.time()
        used = False
        for event in events:
            cx = event.get("x_norm")
            cy = event.get("y_norm")
            ts = float(event.get("timestamp") or 0.0)
            if cx is None or cy is None:
                continue
            if CALIBRATION_MAX_AGE > 0 and ts and now - ts > CALIBRATION_MAX_AGE:
                continue
            x_norm = float(cx)
            y_norm = float(cy)
            source = "calibrated"
            used = True
            logger.info(
                "GOAP: using calibrated dialog target profile=%s scope=%s coords=(%.3f, %.3f)",
                DIALOG_PROFILE,
                DIALOG_SCOPE,
                x_norm,
                y_norm,
            )
            break
        if not used:
            logger.info(
                "GOAP: no calibration for profile=%s scope=%s -> fallback to defaults (%.3f, %.3f)",
                DIALOG_PROFILE,
                DIALOG_SCOPE,
                x_norm,
                y_norm,
            )
        return x_norm, y_norm, source

    def _evaluate_stuck_dialog(self) -> Tuple[Optional[str], int, int]:
        rpc = self._ensure_mem_rpc()
        deaths = recoveries = 0
        reason = None
        if rpc and STUCK_EPISODES_LIMIT > 0:
            payload = {
                "mode": "recent_critical",
                "scope": DIALOG_SCOPE,
                "limit": STUCK_EPISODES_LIMIT,
            }
            response = rpc.query(payload, timeout=0.7) or {}
            entries = response.get("value") or []
            deaths = sum(1 for entry in entries if entry.get("delta") == "hero_dead")
            recoveries = sum(1 for entry in entries if entry.get("delta") == "hero_resurrected")
            if deaths >= STUCK_EPISODES_LIMIT and recoveries == 0:
                reason = f"recent_history_dominates (deaths={deaths})"
        if not reason and STUCK_PLAN_LIMIT > 0 and self.dialog_plans_since_recover >= STUCK_PLAN_LIMIT:
            reason = f"plan_repeat_{self.dialog_plans_since_recover}"
        return reason, deaths, recoveries

    def _apply_jitter(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        if JITTER_RANGE <= 0:
            return x_norm, y_norm
        jitter_x = self.rng.uniform(-JITTER_RANGE, JITTER_RANGE)
        jitter_y = self.rng.uniform(-JITTER_RANGE, JITTER_RANGE)
        return self._clamp_unit(x_norm + jitter_x), self._clamp_unit(y_norm + jitter_y)

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, value))

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
    agent = GOAPAgent()
    agent.run()

if __name__ == "__main__":
    main()
