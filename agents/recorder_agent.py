#!/usr/bin/env python3
import json
import logging
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
ACT_ALIAS = os.getenv("ACT_CMD_ALIAS", "act/request")
TEACHER_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
SNAPSHOT_TOPIC = os.getenv("VISION_SNAPSHOT_TOPIC", "vision/snapshot")
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
REPLAY_TOPIC = os.getenv("REPLAY_STORE_TOPIC", "replay/store")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_SUMMARY_KEY = os.getenv("MEM_EPISODE_KEY", "episodes")
DATA_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
MAX_RECORDS = int(os.getenv("RECORDER_MAX", "1000"))
PIN_THRESHOLD = float(os.getenv("PIN_THRESHOLD", "0.0"))
LEARNING_STAGE = int(os.getenv("LEARNING_STAGE", "1"))
CRITICAL_DELTAS = {"hero_dead", "hero_resurrected"}
CRITICAL_TAG_HINTS = {"critical_dialog", "boss_phase", "hp_low"}
CRITICAL_SCOPE_DEFAULT = os.getenv("CRITICAL_SCOPE_DEFAULT", "critical_dialog:death")
ACTION_CONTEXT_WINDOW_SEC = float(os.getenv("RECORDER_CONTEXT_WINDOW", "2.0"))

# --- Setup ---
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=os.getenv("RECORDER_LOG_LEVEL", "INFO"))
logger = logging.getLogger("recorder_agent")
stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class RecorderAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="recorder_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        self.scene_cache = deque(maxlen=10)
        self.record_count = 0
        self.last_teacher_action = {"source": None, "action": None, "timestamp": 0.0}
        self.last_reward = {"value": 0.0, "timestamp": 0.0}
        self.last_snapshot = None

    def _classify_delta(self, before: dict | None, after: dict | None) -> str:
        if not isinstance(before, dict) or not isinstance(after, dict): return "unknown"
        before_flags, after_flags = (before.get("flags") or {}), (after.get("flags") or {})
        if before_flags.get("death") and not after_flags.get("death"): return "hero_resurrected"
        if after_flags.get("death"): return "hero_dead"
        if set(before.get("text") or []) != set(after.get("text") or []): return "ui_change"
        if len(before.get("objects") or []) != len(after.get("objects") or []): return "object_change"
        return "no_change"

    def _infer_scope(self, scene: dict | None) -> str:
        if not isinstance(scene, dict): return CRITICAL_SCOPE_DEFAULT
        flags, text_blob = scene.get("flags") or {}, " ".join(scene.get("text") or []).lower()
        stats = scene.get("stats") or {}
        if flags.get("death") or "resurrect" in text_blob or "you have died" in text_blob: return "critical_dialog:death"
        if stats.get("hp_pct", 1.0) < 0.2 or "danger" in text_blob: return "combat_high_risk"
        if any(token in text_blob for token in ("boss", "miniboss", "phase")): return "boss_phase"
        return CRITICAL_SCOPE_DEFAULT

    def _is_critical_episode(self, record: dict) -> tuple[bool, str, str, set]:
        delta, tags = record.get("delta"), set(record.get("tags") or [])
        scene_after = record.get("scene_after") or {}
        flags = scene_after.get("flags") or {}
        reason = None
        if delta in CRITICAL_DELTAS: reason = f"delta:{delta}"
        elif flags.get("death"): reason = "flag:death"
        elif tags.intersection(CRITICAL_TAG_HINTS): reason = "tag:critical"
        if reason: tags.add("critical_episode")
        return bool(reason), reason or "", self._infer_scope(scene_after), tags

    def _persist(self, record):
        self.record_count += 1
        path = DATA_DIR / f"sample_{int(record['timestamp'])}_{self.record_count:05d}.json"
        try:
            with path.open("w", encoding="utf-8") as f: json.dump(record, f)
        except IOError as e:
            logger.error("Failed to persist record to %s: %s", path, e)
        return path

    def _publish_transition(self, record):
        if REPLAY_TOPIC:
            transition = {
                "timestamp": record.get("timestamp"),
                "state": record.get("scene"),
                "next_state": record.get("scene_after"),
                "action": record.get("action"),
                "reward": record.get("reward"),
                "teacher": record.get("teacher"),
                "priority": abs(record.get("reward", {}).get("value", 0.0)) + 1e-3,
            }
            self.client.publish(REPLAY_TOPIC, json.dumps(transition))

    def _publish_episode_summary(self, record):
        if not MEM_STORE_TOPIC: return
        scene = record.get("scene", {})
        summary = {
            "timestamp": record.get("timestamp"),
            "goal_id": scene.get("goal_id"),
            "task_id": scene.get("task_id"),
            "action": record.get("action"),
            "teacher": record.get("teacher"),
            "reward": record.get("reward"),
            "text": scene.get("text"),
            "stage": record.get("stage"),
            "delta": record.get("delta"),
        }
        payload = {"op": "episode_summary", "key": MEM_SUMMARY_KEY, "value": summary}
        self.client.publish(MEM_STORE_TOPIC, json.dumps(payload))

    def _prune_if_needed(self):
        files = sorted(DATA_DIR.glob("sample_*.json"))
        if len(files) <= MAX_RECORDS: return
        pruned = 0
        for path in files[:len(files) - MAX_RECORDS]:
            try:
                path.unlink()
                pruned += 1
            except Exception as e:
                logger.warning("Failed to prune file %s: %s", path, e)
                continue
        if pruned:
            self.client.publish("recorder/status", json.dumps({"ok": True, "event": "pruned", "count": len(files), "pruned": pruned, "reason": "max_records"}))

    def _compute_score(self, scene, reward):
        score = float(reward.get("value", 0.0))
        if (reward.get("details") or {}).get("success"): score += 1.0
        if scene.get("goal_status") == "success": score += 1.0
        return score

    def _maybe_pin_episode(self, record, path):
        if not MEM_STORE_TOPIC: return
        scene, reward = record.get("scene", {}), record.get("reward", {})
        score = self._compute_score(scene, reward)
        if score < PIN_THRESHOLD: return
        tags = []
        if scene.get("goal_id"): tags.append(f"goal:{scene.get('goal_id')}")
        if scene.get("mode") in ("sim", "live"): tags.append(scene.get("mode"))
        payload = {
            "op": "pin_candidate",
            "key": "pinned",
            "value": {
                "episode_id": path.name, "timestamp": record.get("timestamp"), "score": score,
                "summary": {"goal_id": scene.get("goal_id"), "task_id": scene.get("task_id"), "reward": reward, "teacher": record.get("teacher"), "stage": record.get("stage"), "delta": record.get("delta")},
                "payload_path": str(path), "tags": tags,
            },
        }
        self.client.publish(MEM_STORE_TOPIC, json.dumps(payload))

    def _maybe_store_critical_episode(self, record, path):
        if not MEM_STORE_TOPIC: return
        is_critical, reason, scope, tags = self._is_critical_episode(record)
        if not is_critical: return
        record.setdefault("tags", []).extend([t for t in tags if t not in record.get("tags", [])])
        payload = {
            "op": "critical_episode",
            "value": {
                "episode_id": path.name, "timestamp": record.get("timestamp"), "delta": record.get("delta"),
                "stage": record.get("stage"), "scope": scope, "reason": reason, "tags": sorted(tags), "payload_path": str(path),
            },
        }
        self.client.publish(MEM_STORE_TOPIC, json.dumps(payload))

    def _on_connect(self, client, userdata, flags, rc):
        topics = {(SCENE_TOPIC, 0), (ACT_TOPIC, 0), (SNAPSHOT_TOPIC, 0), (FRAME_TOPIC, 0)}
        if TEACHER_TOPIC: topics.add((TEACHER_TOPIC, 0))
        if REWARD_TOPIC: topics.add((REWARD_TOPIC, 0))
        if ACT_ALIAS and ACT_ALIAS != ACT_TOPIC: topics.add((ACT_ALIAS, 0))
        
        if _as_int(rc) == 0:
            client.subscribe(list(topics))
            client.publish("recorder/status", json.dumps({"ok": True, "event": "recorder_ready", "path": str(DATA_DIR)}))
            logger.info("Recorder connected")
        else:
            client.publish("recorder/status", json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))
            logger.error("Recorder connect failed rc=%s", _as_int(rc))

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: data = {"raw": msg.payload}

        if msg.topic == SCENE_TOPIC and data.get("ok"):
            self.scene_cache.append((time.time(), data))
        elif msg.topic in (SNAPSHOT_TOPIC, FRAME_TOPIC) and data.get("ok"):
            self.last_snapshot = data
        elif msg.topic == TEACHER_TOPIC:
            self.last_teacher_action = {
                "source": data.get("source") if isinstance(data, dict) else None,
                "action": data.get("action") if isinstance(data, dict) else msg.payload,
                "timestamp": time.time(),
                "reasoning": data.get("reasoning") if isinstance(data, dict) else None,
            }
        elif msg.topic == REWARD_TOPIC:
            try: val = float(data.get("reward") if isinstance(data, dict) else 0.0)
            except (TypeError, ValueError): val = 0.0
            self.last_reward = {"value": val, "timestamp": time.time(), "details": data}
        elif msg.topic in (ACT_TOPIC, ACT_ALIAS):
            if not self.scene_cache: return
            now = time.time()
            ts, scene_after = self.scene_cache[-1]
            scene_before = self.scene_cache[-2][1] if len(self.scene_cache) > 1 else scene_after
            
            # Inject snapshot if recent
            if self.last_snapshot and now - self.last_snapshot.get("timestamp", 0) < 2.0:
                scene_before["image_b64"] = self.last_snapshot.get("image_b64")

            delta = self._classify_delta(scene_before, scene_after)
            
            teacher_action = self.last_teacher_action if now - self.last_teacher_action.get("timestamp", 0) < ACTION_CONTEXT_WINDOW_SEC else {}
            reward = self.last_reward if now - self.last_reward.get("timestamp", 0) < ACTION_CONTEXT_WINDOW_SEC else {}

            record = {
                "timestamp": now, "scene_time": ts, "scene": scene_before, "scene_after": scene_after,
                "action": data, "teacher": teacher_action, "reward": reward, "stage": LEARNING_STAGE, "delta": delta,
            }
            if LEARNING_STAGE == 0: record.setdefault("tags", []).append("stage0_exploration")
            
            episode_path = self._persist(record)
            self._publish_transition(record)
            self._publish_episode_summary(record)
            self._maybe_pin_episode(record, episode_path)
            self._maybe_store_critical_episode(record, episode_path)
            self._prune_if_needed()

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Recorder agent shut down")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    RecorderAgent().run()

if __name__ == "__main__":
    main()
