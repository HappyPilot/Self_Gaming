#!/usr/bin/env python3
import json
import logging
import os
import signal
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

from utils.frame_transport import get_frame_bytes
from utils.latency import emit_control_metric

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
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC", "vision/objects")
OCR_TEXT_TOPIC = os.getenv("OCR_TEXT_TOPIC", "ocr/text")
OCR_EASY_TOPIC = os.getenv("OCR_EASY_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
PERCEPTION_TOPIC = os.getenv("PERCEPTION_TOPIC", "vision/observation")
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
RECORDER_SESSION_ENABLE = os.getenv("RECORDER_SESSION_ENABLE", "1") == "1"
RECORDER_DATASET_DIR = Path(os.getenv("RECORDER_DATASET_DIR", "/mnt/ssd/datasets"))
RECORDER_FRAME_TOPIC = os.getenv("RECORDER_FRAME_TOPIC") or FRAME_TOPIC
RECORDER_SESSION_ID = os.getenv("RECORDER_SESSION_ID", "")
RECORDER_GAME_ID = os.getenv("RECORDER_GAME_ID") or os.getenv("GAME_ID", "default")
RECORDER_GAME_ID = RECORDER_GAME_ID.strip() or "default"
RECORDER_EXPECT_FPS = float(os.getenv("RECORDER_EXPECT_FPS", "0"))

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

def _parse_topics(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}

def _build_sensor_topics() -> set[str]:
    defaults = [SCENE_TOPIC, OBJECT_TOPIC, OCR_TEXT_TOPIC, OCR_EASY_TOPIC, SIMPLE_OCR_TOPIC, PERCEPTION_TOPIC]
    if TEACHER_TOPIC:
        defaults.append(TEACHER_TOPIC)
    if REWARD_TOPIC:
        defaults.append(REWARD_TOPIC)
    raw = os.getenv("RECORDER_SENSOR_TOPICS")
    if raw:
        return _parse_topics(raw)
    return set(defaults)

SENSOR_TOPICS = _build_sensor_topics()


class RecorderQC:
    def __init__(self, expected_fps: float) -> None:
        self.expected_fps = expected_fps
        self.expected_period = 1.0 / expected_fps if expected_fps > 0 else 0.0
        self.frames_total = 0
        self.frames_dropped = 0
        self.last_frame_ts = None
        self.action_total = 0
        self.last_action_ts = None
        self.action_lag_ms = deque(maxlen=5000)
        self.move_sum_sq = 0.0
        self.move_count = 0
        self.button_count = 0

    def record_frame(self, ts: float) -> None:
        self.frames_total += 1
        if self.last_frame_ts is not None and self.expected_period > 0:
            gap = ts - self.last_frame_ts
            if gap > self.expected_period * 1.5:
                dropped = max(0, int(gap / self.expected_period) - 1)
                self.frames_dropped += dropped
        self.last_frame_ts = ts

    def record_action(self, ts: float, action: dict | None) -> None:
        self.action_total += 1
        self.last_action_ts = ts
        if self.last_frame_ts is not None:
            lag_ms = max(0.0, (ts - self.last_frame_ts) * 1000.0)
            self.action_lag_ms.append(lag_ms)
        if isinstance(action, dict):
            dx = action.get("dx")
            dy = action.get("dy")
            if isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
                self.move_sum_sq += float(dx) ** 2 + float(dy) ** 2
                self.move_count += 1
            label = str(action.get("label") or action.get("action") or "").lower()
            if label.startswith("key_") or label in {"key_press", "click_primary", "click_secondary", "mouse_click"}:
                self.button_count += 1

    def summary(self, start_ts: float, end_ts: float) -> dict:
        duration = max(0.001, end_ts - start_ts)
        dropped_pct = None
        if self.expected_period > 0 and self.frames_total > 0:
            dropped_pct = round((self.frames_dropped / float(self.frames_total)) * 100.0, 2)
        lag_mean = None
        lag_p95 = None
        lag_max = None
        if self.action_lag_ms:
            lag_values = sorted(self.action_lag_ms)
            lag_mean = round(sum(lag_values) / float(len(lag_values)), 3)
            lag_max = round(max(lag_values), 3)
            idx = int(0.95 * (len(lag_values) - 1))
            lag_p95 = round(lag_values[idx], 3)
        jitter_rms = None
        if self.move_count > 0:
            jitter_rms = round((self.move_sum_sq / float(self.move_count)) ** 0.5, 3)
        button_rate = round(self.button_count / duration, 3) if self.button_count > 0 else 0.0
        return {
            "duration_sec": round(duration, 3),
            "frames_total": self.frames_total,
            "frames_dropped": self.frames_dropped,
            "dropped_frames_pct": dropped_pct,
            "input_lag_ms_mean": lag_mean,
            "input_lag_ms_p95": lag_p95,
            "input_lag_ms_max": lag_max,
            "stick_jitter_rms": jitter_rms,
            "button_spam_rate_hz": button_rate,
        }


class RecordingSession:
    def __init__(self) -> None:
        self.enabled = RECORDER_SESSION_ENABLE
        self.session_id = None
        self.start_ts = time.time()
        self.frames_written = 0
        self.actions_written = 0
        self.sensors_written = 0
        self.qc = RecorderQC(RECORDER_EXPECT_FPS)
        if not self.enabled:
            self.base_dir = None
            return
        session_id = RECORDER_SESSION_ID or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_id = session_id
        self.base_dir = RECORDER_DATASET_DIR / RECORDER_GAME_ID / session_id
        self.frames_dir = self.base_dir / "frames"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.actions_fp = (self.base_dir / "actions.jsonl").open("a", encoding="utf-8", buffering=1)
        self.sensors_fp = (self.base_dir / "sensors.jsonl").open("a", encoding="utf-8", buffering=1)
        meta = {
            "session_id": session_id,
            "game_id": RECORDER_GAME_ID,
            "started_at": self.start_ts,
            "frame_topic": RECORDER_FRAME_TOPIC,
            "sensor_topics": sorted(SENSOR_TOPICS),
            "action_topics": [ACT_TOPIC] + ([ACT_ALIAS] if ACT_ALIAS else []),
            "profile": os.getenv("SG_PROFILE"),
            "expected_fps": RECORDER_EXPECT_FPS,
        }
        (self.base_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _write_jsonl(self, fp, payload: dict) -> None:
        fp.write(json.dumps(payload) + "\n")

    def record_frame(self, payload: dict | None) -> None:
        if not self.enabled or payload is None:
            return
        data = get_frame_bytes(payload)
        if not data:
            return
        ts = float(payload.get("timestamp", time.time()))
        filename = f"{int(ts * 1000)}_{self.frames_written:06d}.jpg"
        try:
            (self.frames_dir / filename).write_bytes(data)
            self.frames_written += 1
            self.qc.record_frame(ts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write frame: %s", exc)

    def record_action(self, action: dict | None, ts: float, topic: str) -> None:
        if not self.enabled:
            return
        payload = {"timestamp": ts, "topic": topic, "action": action}
        try:
            self._write_jsonl(self.actions_fp, payload)
            self.actions_written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write action: %s", exc)
        self.qc.record_action(ts, action)

    def record_sensor(self, topic: str, payload: dict | None, ts: float) -> None:
        if not self.enabled:
            return
        data = {"timestamp": ts, "topic": topic, "payload": payload}
        try:
            self._write_jsonl(self.sensors_fp, data)
            self.sensors_written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write sensor: %s", exc)

    def close(self) -> dict | None:
        if not self.enabled or self.base_dir is None:
            self.enabled = False
            return None
        self.enabled = False
        end_ts = time.time()
        qc = self.qc.summary(self.start_ts, end_ts)
        qc.update({
            "frames_written": self.frames_written,
            "actions_written": self.actions_written,
            "sensors_written": self.sensors_written,
        })
        try:
            (self.base_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write qc.json: %s", exc)
        try:
            self.actions_fp.close()
            self.sensors_fp.close()
        except Exception:
            pass
        return qc

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
        self.session = RecordingSession()

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
        topics = {(SCENE_TOPIC, 0), (ACT_TOPIC, 0), (SNAPSHOT_TOPIC, 0), (RECORDER_FRAME_TOPIC, 0)}
        for topic in SENSOR_TOPICS:
            topics.add((topic, 0))
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

        if msg.topic == RECORDER_FRAME_TOPIC and isinstance(data, dict):
            self.session.record_frame(data)
        if msg.topic in SENSOR_TOPICS and isinstance(data, dict):
            ts = float(data.get("timestamp", time.time()))
            self.session.record_sensor(msg.topic, data, ts)

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
            now = time.time()
            self.session.record_action(data if isinstance(data, dict) else None, now, msg.topic)
            if not self.scene_cache: return
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

    def _publish_qc_metrics(self, qc: dict | None) -> None:
        if not qc:
            return
        tags = {"agent": "recorder_agent", "game_id": RECORDER_GAME_ID}
        if self.session.session_id:
            tags["session_id"] = self.session.session_id
        for key, value in qc.items():
            if isinstance(value, (int, float)):
                emit_control_metric(self.client, f"recorder.{key}", value, tags=tags)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        qc = self.session.close()
        self._publish_qc_metrics(qc)
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
