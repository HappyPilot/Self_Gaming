#!/usr/bin/env python3
import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt

# --- Setup ---
logging.basicConfig(level=os.getenv("MEM_LOG_LEVEL", "INFO"))
logger = logging.getLogger("mem_agent")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_QUERY_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query")
MEM_REPLY_TOPIC = os.getenv("MEM_REPLY_TOPIC", "mem/reply")
MEM_RESPONSE_TOPIC = os.getenv("MEM_RESPONSE_TOPIC", "mem/response")

PINNED_PATH = Path(os.getenv("MEM_PINNED_PATH", "/mnt/ssd/memory/pinned.json"))
RULES_PATH = Path(os.getenv("MEM_RULES_PATH", "/mnt/ssd/memory/rules.json"))
CALIBRATION_PATH = Path(os.getenv("MEM_CALIBRATION_PATH", "/mnt/ssd/memory/calibration_events.json"))

MAX_PINNED = int(os.getenv("MAX_PINNED_EPISODES", "100"))
MAX_RULES = int(os.getenv("MEM_RULES_MAX", "200"))
RECENT_CRITICAL_MAX = int(os.getenv("MEM_RECENT_CRITICAL_MAX", "64"))
CALIBRATION_MAX = int(os.getenv("MEM_CALIBRATION_MAX", "200"))
RULE_DECAY_SEC = float(os.getenv("MEM_RULE_DECAY_SEC", "7200"))
RULE_DECAY_PENALTY = float(os.getenv("MEM_RULE_DECAY_PENALTY", "0.05"))
RULE_RETIRE_THRESHOLD = float(os.getenv("MEM_RULE_RETIRE_THRESHOLD", "0.1"))

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class MemAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="mem_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        self.kv_store = {}
        self.vector_notes = defaultdict(list)
        self.episode_summaries = deque(maxlen=int(os.getenv("MEM_EPISODE_MAX", "500")))
        self.pinned_episodes = []
        self.rules = []
        self.recent_critical = deque(maxlen=RECENT_CRITICAL_MAX)
        self.calibration_events = deque(maxlen=CALIBRATION_MAX)
        
        self._load_persistence()

    def _load_persistence(self):
        self._load_json(PINNED_PATH, self.pinned_episodes)
        self._load_rules()
        self._load_calibration()

    def _load_json(self, path: Path, target_list: list):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    target_list.extend(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

    def _load_rules(self):
        if RULES_PATH.exists():
            try:
                data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data:
                        self.rules.append(self._normalize_rule(entry))
            except Exception as e:
                logger.warning("Failed to load rules: %s", e)

    def _load_calibration(self):
        if CALIBRATION_PATH.exists():
            try:
                data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data[-CALIBRATION_MAX:]:
                        self.calibration_events.append(entry)
            except Exception as e:
                logger.warning("Failed to load calibration: %s", e)

    def _save_persistence(self):
        self._save_json(PINNED_PATH, self.pinned_episodes)
        self._save_json(RULES_PATH, self.rules)
        self._save_json(CALIBRATION_PATH, list(self.calibration_events))

    def _save_json(self, path: Path, data: list):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save %s: %s", path, e)

    def _normalize_rule(self, entry: dict) -> dict:
        entry = dict(entry)
        entry.setdefault("usage_count", 0)
        entry.setdefault("last_used_at", entry.get("created_at", time.time()))
        entry.setdefault("last_decay", entry.get("created_at", time.time()))
        entry.setdefault("retired", False)
        return entry

    def _apply_rule_decay(self, now: Optional[float] = None):
        if RULE_DECAY_SEC <= 0: return
        now = now or time.time()
        changed = False
        for entry in self.rules:
            last_decay = entry.get("last_decay") or entry.get("created_at", now)
            last_used = entry.get("last_used_at") or entry.get("created_at", now)
            if min(now - last_decay, now - last_used) >= RULE_DECAY_SEC and not entry.get("retired"):
                entry["confidence"] = max(0.0, float(entry.get("confidence", 0.0)) - RULE_DECAY_PENALTY)
                entry["last_decay"] = now
                if entry["confidence"] <= RULE_RETIRE_THRESHOLD:
                    entry["retired"] = True
                changed = True
        if changed:
            self._save_json(RULES_PATH, self.rules)

    def _insert_rule(self, rule):
        text = (rule.get("text") or "").strip()
        if not text: return
        entry = {
            "rule_id": rule.get("rule_id") or f"rule_{int(time.time() * 1000)}",
            "scope": (rule.get("scope") or "generic_ui").lower(),
            "text": text,
            "source": rule.get("source") or "reflection",
            "confidence": float(rule.get("confidence", 0.5)),
            "created_at": rule.get("created_at") or time.time(),
            "usage_count": int(rule.get("usage_count", 0)),
            "retired": bool(rule.get("retired", False))
        }
        entry["last_used_at"] = rule.get("last_used_at") or entry["created_at"]
        entry["last_decay"] = rule.get("last_decay") or entry["created_at"]
        
        existing = next((r for r in self.rules if r.get("rule_id") == entry["rule_id"]), None)
        if existing:
            self.rules[self.rules.index(existing)] = entry
        else:
            for idx, current in enumerate(self.rules):
                if current.get("text") == entry["text"] and current.get("scope") == entry["scope"]:
                    self.rules[idx] = entry
                    break
            else:
                self.rules.append(entry)
        self._apply_rule_decay(time.time())
        self.rules.sort(key=lambda r: (-float(r.get("confidence", 0.0)), -float(r.get("created_at", 0.0))))
        while len(self.rules) > MAX_RULES: self.rules.pop()
        self._save_json(RULES_PATH, self.rules)

    def _store(self, data):
        op, key, req_id = data.get("op"), data.get("key"), data.get("request_id")
        if op == "set": self.kv_store[key] = data.get("value")
        elif op == "append": self.kv_store.setdefault(key, []).append(data.get("value"))
        elif op == "vector_append": self.vector_notes[key].append(data.get("value"))
        elif op == "episode_summary":
            if data.get("value"): self.episode_summaries.append({"key": key, "summary": data.get("value")})
        elif op == "pin_candidate":
            val = data.get("value")
            if val:
                self.pinned_episodes.append(val)
                self.pinned_episodes.sort(key=lambda e: (-float(e.get("score", 0.0)), -float(e.get("timestamp", 0.0))))
                if len(self.pinned_episodes) > MAX_PINNED: self.pinned_episodes.pop()
                self._save_json(PINNED_PATH, self.pinned_episodes)
        elif op == "rule_insert":
            if data.get("value"): self._insert_rule(data.get("value"))
        elif op == "critical_episode":
            val = data.get("value")
            if val and val.get("episode_id"):
                for ex in list(self.recent_critical):
                    if ex.get("episode_id") == val.get("episode_id"): self.recent_critical.remove(ex); break
                self.recent_critical.append(val)
        elif op == "rule_used":
            val = data.get("value") or {}
            rid, ts = val.get("rule_id"), val.get("timestamp") or time.time()
            for r in self.rules:
                if r.get("rule_id") == rid:
                    r["usage_count"] = int(r.get("usage_count", 0)) + 1
                    r["last_used_at"] = ts
                    r["retired"] = False
                    r["confidence"] = min(1.0, r.get("confidence", 0.5) + RULE_DECAY_PENALTY)
                    self._save_json(RULES_PATH, self.rules)
                    break
        elif op == "calibration_success":
            val = data.get("value")
            if val:
                self.calibration_events.append(val)
                self._save_json(CALIBRATION_PATH, list(self.calibration_events))
        
        res = {"ok": True, "event": "mem_stored", "key": key, "op": op, "timestamp": time.time()}
        if req_id: res["request_id"] = req_id
        return res

    def _query(self, data):
        key, mode, req_id = data.get("key"), data.get("mode", "kv"), data.get("request_id")
        if mode == "vector": value = self.vector_notes.get(key, [])
        elif mode == "episodes": value = list(self.episode_summaries)
        elif mode == "pinned":
            limit, tag, scope = int(data.get("limit", 10)), data.get("tag"), data.get("scope")
            value = [ep for ep in self.pinned_episodes if (not tag or tag in (ep.get("tags") or [])) and (not scope or ep.get("scope") == scope or scope in (ep.get("tags") or []))][:limit]
        elif mode == "rules":
            self._apply_rule_decay()
            limit, scope = int(data.get("limit", 5)), (data.get("scope") or "").lower()
            include_retired = data.get("include_retired") in {True, "true", "1"}
            value = [r for r in self.rules if (not scope or r.get("scope") == scope) and (include_retired or not r.get("retired"))][:limit]
        elif mode == "recent_critical":
            limit, scope, tag = int(data.get("limit", 10)), data.get("scope"), data.get("tag")
            subset = list(self.recent_critical); subset.reverse()
            value = [e for e in subset if (not scope or e.get("scope") == scope) and (not tag or tag in (e.get("tags") or []))][:limit]
        elif mode == "calibration_events":
            limit, scope, profile = int(data.get("limit", 10)), data.get("scope"), (data.get("profile") or "").strip().lower()
            subset = list(self.calibration_events); subset.reverse()
            value = [e for e in subset if (not scope or e.get("scope") == scope) and (not profile or (e.get("profile") or "") == profile)][:limit]
        else: value = self.kv_store.get(key)
        
        res = {"ok": True, "event": "mem_result", "key": key, "value": value, "mode": mode}
        if req_id: res["request_id"] = req_id
        return res

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(MEM_STORE_TOPIC, 0), (MEM_QUERY_TOPIC, 0)])
            client.publish(MEM_REPLY_TOPIC, json.dumps({"ok": True, "event": "mem_agent_ready"}))
            logger.info("Mem agent connected.")
        else:
            client.publish(MEM_REPLY_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_message(self, client, userdata, msg):
        try: data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: data = {"raw": msg.payload}
        if msg.topic == MEM_STORE_TOPIC:
            client.publish(MEM_REPLY_TOPIC, json.dumps(self._store(data)))
        elif msg.topic == MEM_QUERY_TOPIC:
            try:
                res = self._query(data)
                client.publish(MEM_RESPONSE_TOPIC, json.dumps(res))
            except Exception as e:
                logger.error("Query failed: %s", e)
                client.publish(MEM_RESPONSE_TOPIC, json.dumps({"ok": False, "error": str(e), "request_id": data.get("request_id")}))

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self._save_persistence()
        logger.info("Mem agent shut down.")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received, shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = MemAgent()
    agent.run()

if __name__ == "__main__":
    main()
