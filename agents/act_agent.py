#!/usr/bin/env python3
import json
import os
import queue
import signal
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt
from control_profile import load_profile, safe_profile

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACT_CMD_ALIAS = os.getenv("ACT_CMD_ALIAS", "act/request")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
ACT_RESULT_ALIAS = os.getenv("ACT_RESULT_ALIAS", "act/feedback")
ACTION_QUEUE_MAX = int(os.getenv("ACTION_QUEUE_MAX", "10"))
ACT_LOG_TOPIC = os.getenv("ACT_LOG_TOPIC", "act/log")
ACTION_LOG_PATH = os.getenv("ACTION_LOG_PATH", "/app/logs/actions.log")
GAME_SCHEMA_TOPIC = os.getenv("GAME_SCHEMA_TOPIC", "game/schema")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
CONTROL_PROFILE_PATH = os.getenv("CONTROL_PROFILE_PATH", "/app/data/control_profiles.json")

# Rate-limit defaults (overridden by profile if present)
DEFAULT_WINDOW_SEC = float(os.getenv("ACT_WINDOW_SEC", "10.0"))
DEFAULT_MAX_ACTIONS = int(os.getenv("ACT_MAX_ACTIONS", "6"))

# File logger for actions
def _append_log(line: str):
    if not ACTION_LOG_PATH:
        return
    try:
        os.makedirs(os.path.dirname(ACTION_LOG_PATH), exist_ok=True)
        with open(ACTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# --- Setup ---
stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class ActAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="act_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.action_queue = queue.Queue(maxsize=ACTION_QUEUE_MAX)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.profile = safe_profile()
        self.profile_loaded_at = 0.0
        self.recent_actions: deque = deque(maxlen=200)
        self.rate_window: deque = deque()
        self.state_history: deque = deque(maxlen=200)  # (timestamp, state)
        self.pending_evals: deque = deque(maxlen=50)   # actions awaiting post-state

    def _publish_act(self, payload):
        for topic in {ACT_RESULT_TOPIC, ACT_RESULT_ALIAS}:
            if topic: self.client.publish(topic, json.dumps(payload))
        if ACT_LOG_TOPIC:
            self.client.publish(ACT_LOG_TOPIC, json.dumps(payload))

    def _apply_action(self, action):
        pre_state = self._latest_state()
        ts = time.time()
        # Placeholder for real device interactions
        time.sleep(0.1)
        if pre_state is not None:
            try:
                self.pending_evals.append({"timestamp": ts, "action": action, "pre_state": pre_state})
            except Exception:
                pass
        return {"ok": True, "applied": action, "timestamp": time.time()}

    def _latest_state(self) -> dict | None:
        if not self.state_history:
            return None
        return self.state_history[-1][1]

    def _maybe_eval_pending(self, state_ts: float, state: dict):
        if not self.pending_evals:
            return
        evaluated = []
        for item in list(self.pending_evals):
            act_ts = item.get("timestamp", 0)
            if state_ts <= act_ts:
                continue
            pre = item.get("pre_state") or {}
            eval_payload = self._eval_delta(item.get("action", {}), pre, state, act_ts, state_ts)
            self._publish_act(eval_payload)
            evaluated.append(item)
        for item in evaluated:
            try:
                self.pending_evals.remove(item)
            except ValueError:
                pass

    def _eval_delta(self, action: dict, pre: dict, post: dict, ts_action: float, ts_state: float) -> dict:
        def _enemy_count(st):
            stats = st.get("stats") or {}
            if isinstance(stats, dict) and "enemy_count" in stats:
                try:
                    return float(stats.get("enemy_count") or 0)
                except Exception:
                    return 0.0
            objs = st.get("objects") or []
            return float(len(objs)) if isinstance(objs, list) else 0.0

        def _mean_bright(st):
            if "mean" in st:
                return float(st.get("mean") or 0.0)
            if "mean_brightness" in st:
                return float(st.get("mean_brightness") or 0.0)
            return 0.0

        def _text_hash(st):
            texts = st.get("text") or []
            if isinstance(texts, list):
                joined = " ".join(str(t) for t in texts)
            else:
                joined = str(texts)
            return hash(joined)

        pre_enemy = _enemy_count(pre)
        post_enemy = _enemy_count(post)
        pre_mean = _mean_bright(pre)
        post_mean = _mean_bright(post)
        text_changed = _text_hash(pre) != _text_hash(post)

        return {
            "ok": True,
            "event": "action_eval",
            "action": action,
            "ts_action": ts_action,
            "ts_state": ts_state,
            "delta_enemy": post_enemy - pre_enemy,
            "delta_mean": post_mean - pre_mean,
            "text_changed": text_changed,
            "pre_enemy": pre_enemy,
            "post_enemy": post_enemy,
        }

    # Profile ---------------------------------------------------------------
    def _load_profile(self, game_id: str | None):
        if not game_id:
            return
        path = Path(CONTROL_PROFILE_PATH) if CONTROL_PROFILE_PATH else None
        loaded = load_profile(game_id, path=path) if path else load_profile(game_id)
        if loaded:
            self.profile = loaded
            self.profile_loaded_at = time.time()
            self._publish_act({"ok": True, "event": "profile_loaded", "game_id": game_id})

    def _update_profile_from_schema(self, payload: dict):
        if not isinstance(payload, dict):
            return
        schema = payload.get("schema") if "schema" in payload else payload
        if not isinstance(schema, dict):
            return
        game_id = str(schema.get("game_id") or "unknown_game")
        profile = schema.get("profile")
        if isinstance(profile, dict):
            self.profile = profile
            self.profile_loaded_at = time.time()
            self._publish_act({"ok": True, "event": "profile_received", "game_id": game_id})
        else:
            self._load_profile(game_id)

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            topics = [(t, 0) for t in {ACT_CMD_TOPIC, ACT_CMD_ALIAS, GAME_SCHEMA_TOPIC, SCENE_TOPIC} if t]
            for topic in topics: client.subscribe(topic)
            self._publish_act({"ok": True, "event": "act_agent_ready"})
        else:
            self._publish_act({"ok": False, "event": "connect_failed", "code": _as_int(rc)})

    def _on_message(self, client, userdata, msg):
        if msg.topic == SCENE_TOPIC:
            try:
                state = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            if isinstance(state, dict):
                ts = float(state.get("timestamp", time.time()))
                self.state_history.append((ts, state))
                self._maybe_eval_pending(ts, state)
            return
        if msg.topic == GAME_SCHEMA_TOPIC:
            try:
                payload = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception:
                return
            self._update_profile_from_schema(payload)
            return
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
            if not isinstance(data, dict) or "action" not in data:
                raise ValueError("Invalid action payload")
            self.action_queue.put_nowait(data)
        except queue.Full:
            self._publish_act({"ok": False, "error": "action_queue_full", "received": data})
        except Exception:
            self._publish_act({"ok": False, "error": "invalid_action_payload", "received": msg.payload.decode("utf-8", "ignore")})

    # Filtering -------------------------------------------------------------
    def _within_rate_limit(self, now: float) -> bool:
        window_sec = float(self.profile.get("window_sec", DEFAULT_WINDOW_SEC) or DEFAULT_WINDOW_SEC)
        max_actions = int(self.profile.get("max_actions_per_window", DEFAULT_MAX_ACTIONS) or DEFAULT_MAX_ACTIONS)
        self.rate_window.append(now)
        while self.rate_window and now - self.rate_window[0] > window_sec:
            self.rate_window.popleft()
        return len(self.rate_window) <= max_actions

    def _is_allowed(self, action: dict) -> tuple[bool, str]:
        kind = action.get("action")
        if kind == "dump_recent":
            return True, "dump"
        if kind == "wait":
            return True, "ok"
        if kind == "mouse_move":
            if not self.profile.get("allow_mouse_move", True):
                return False, "blocked_mouse_move"
            return True, "ok"
        if kind == "click_primary":
            if not self.profile.get("allow_primary", False):
                return False, "blocked_primary"
            return True, "ok"
        if kind == "click_secondary":
            if not self.profile.get("allow_secondary", False):
                return False, "blocked_secondary"
            return True, "ok"
        if kind == "key_press":
            key = str(action.get("key") or "").lower()
            if not key:
                return False, "invalid_key"
            forbidden = {k.lower() for k in self.profile.get("forbidden_keys", [])}
            if key in forbidden:
                return False, "blocked_forbidden_key"
            allowed = self.profile.get("allowed_keys")
            if allowed is None:
                allowed = []
            if allowed and key not in {k.lower() for k in allowed}:
                return False, "blocked_not_whitelisted"
            if not allowed:
                return False, "blocked_no_keys_allowed"
            return True, "ok"
        return False, "unknown_action"

    def _worker_loop(self):
        while not stop_event.is_set():
            try:
                action_data = self.action_queue.get(timeout=0.5)
                allowed, reason = self._is_allowed(action_data)
                now = time.time()
                if not allowed:
                    blocked = {"ok": False, "blocked": True, "reason": reason, "action": action_data, "timestamp": now}
                    self._publish_act(blocked)
                    _append_log(f"{now:.3f} blocked={reason} action={action_data}")
                    continue
                if reason == "dump":
                    snapshot = list(self.recent_actions)[-30:]
                    self._publish_act({"ok": True, "event": "recent_actions", "items": snapshot, "timestamp": now})
                    continue
                if action_data.get("action") != "mouse_move" and not self._within_rate_limit(now):
                    blocked = {"ok": False, "blocked": True, "reason": "rate_limited", "action": action_data, "timestamp": now}
                    self._publish_act(blocked)
                    _append_log(f"{now:.3f} blocked=rate_limited action={action_data}")
                    continue
                result = self._apply_action(action_data)
                self.recent_actions.append(action_data)
                _append_log(f"{now:.3f} action={action_data}")
                self._publish_act(result)
            except queue.Empty:
                continue
            except Exception as e:
                self._publish_act({"ok": False, "error": f"action_failed: {e}"})

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.worker_thread.start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()

def _handle_signal(signum, frame):
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = ActAgent()
    agent.run()

if __name__ == "__main__":
    main()
