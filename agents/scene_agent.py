#!/usr/bin/env python3
"""Scene agent that fuses raw perception topics into a high-level scene."""
import json
import os
import signal
import threading
import time
from collections import deque
from typing import Dict

import paho.mqtt.client as mqtt

from utils.latency import emit_latency, get_sla_ms
# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
VISION_MEAN_TOPIC = os.getenv("VISION_MEAN_TOPIC", "vision/mean")
VISION_SNAPSHOT_TOPIC = os.getenv("VISION_SNAPSHOT_TOPIC", "vision/snapshot")
OBJECT_TOPIC = os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
OBSERVATION_TOPIC = os.getenv("VISION_OBSERVATION_TOPIC", "")
OCR_TEXT_TOPIC = os.getenv("OCR_TEXT_TOPIC", "ocr/text")
OCR_EASY_TOPIC = os.getenv("OCR_EASY_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
WINDOW_SEC = float(os.getenv("SCENE_WINDOW_SEC", "2.0"))
NORMALIZE_TEXT = os.getenv("SCENE_NORMALIZE_TEXT", "1") != "0"
TEXT_TRANSLATION = str.maketrans({"Я": "R", "я": "r", "С": "C", "с": "c", "Н": "H", "н": "h", "К": "K", "к": "k", "Т": "T", "т": "t", "А": "A", "а": "a", "В": "B", "в": "b", "Е": "E", "е": "e", "М": "M", "м": "m", "О": "O", "о": "o", "Р": "P", "р": "p", "Ь": "b", "Ы": "y", "Л": "L", "л": "l", "Д": "D", "д": "d"})
DEATH_KEYWORDS = [kw.strip().lower() for kw in os.getenv("SCENE_DEATH_KEYWORDS", "you have died,resurrect,revive,respawn,resurrect in town,checkpoint").split(",") if kw.strip()]
DEATH_SYMBOLS = [sym.strip() for sym in os.getenv("SCENE_DEATH_SYMBOLS", "*,†,+,☠").split(",") if sym.strip()]
DEATH_SYMBOL_LINES = int(os.getenv("SCENE_DEATH_SYMBOL_LINES", "1"))
DEATH_SYMBOL_PERSIST = float(os.getenv("SCENE_DEATH_SYMBOL_PERSIST", "1.0"))
DEATH_SYMBOL_MEAN_THRESHOLD = float(os.getenv("SCENE_DEATH_SYMBOL_MEAN_THRESHOLD", "0.28"))
DEATH_OBJECT_THRESHOLD = int(os.getenv("SCENE_DEATH_OBJECT_THRESHOLD", "2"))
PLAYER_LABELS = {label.strip().lower() for label in os.getenv("SCENE_PLAYER_LABELS", "player,person,hero,character,avenger").split(",") if label.strip()}
ENEMY_KEYWORDS = {label.strip().lower() for label in os.getenv("SCENE_ENEMY_KEYWORDS", "enemy,boss,monster,bandit,warrior,archer,necromancer").split(",") if label.strip()}
RESOURCE_KEYWORDS = {"life": ["life", "hp", "health"], "mana": ["mana", "mp"]}
ALLOW_GENERIC_PLAYER = os.getenv("SCENE_GENERIC_PLAYER", "1") != "0"
ALLOW_GENERIC_ENEMIES = os.getenv("SCENE_GENERIC_ENEMIES", "1") != "0"
SLA_STAGE_FUSE_MS = get_sla_ms("SLA_STAGE_FUSE_MS")

stop_event = threading.Event()

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0

class SceneAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="scene_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.state = {
            "mean": deque(maxlen=10), "easy_text": "", "simple_text": "", "snapshot_ts": 0.0,
            "objects": [], "objects_ts": 0.0, "text_zones": {}, "observation": {}, "observation_ts": 0.0,
        }
        self._symbolic_candidate_since = 0.0

    def _symbolic_text_only(self, entries):
        lines = 0
        for entry in entries:
            cleaned = entry.strip()
            if not cleaned: continue
            if any(ch.isalpha() for ch in cleaned): return False
            if DEATH_SYMBOLS and any(sym in cleaned for sym in DEATH_SYMBOLS):
                lines += 1
            else:
                return False
        return lines >= DEATH_SYMBOL_LINES

    def _normalize_text(self, entry: str) -> str:
        return entry.translate(TEXT_TRANSLATION) if isinstance(entry, str) else entry

    def _object_matches(self, objects):
        for obj in objects:
            label = str(obj.get("label") or obj.get("text") or obj.get("class") or "").lower()
            if label and any(kw in label for kw in DEATH_KEYWORDS):
                return True
        return False

    def _normalize_bbox(self, bbox):
        if not bbox or len(bbox) != 4: return None
        return [round(float(c), 4) for c in bbox]

    def _extract_player(self, objects):
        best, fallback = None, None
        best_score, fallback_score = float("inf"), float("inf")
        for obj in objects:
            bbox = obj.get("bbox") or obj.get("box")
            if not bbox: continue
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            dist = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
            label = str(obj.get("label") or "").lower()
            if not PLAYER_LABELS or label in PLAYER_LABELS:
                if dist < best_score:
                    best_score, best = dist, {"label": obj.get("label"), "confidence": obj.get("confidence"), "bbox": self._normalize_bbox(bbox)}
            elif ALLOW_GENERIC_PLAYER and dist < fallback_score:
                fallback_score, fallback = dist, {"label": obj.get("label"), "confidence": obj.get("confidence"), "bbox": self._normalize_bbox(bbox)}
        return best or fallback

    def _extract_enemies(self, objects):
        matches = []
        for obj in objects:
            bbox = obj.get("bbox") or obj.get("box")
            if not bbox: continue
            label = str(obj.get("label") or "").lower()
            if not ENEMY_KEYWORDS or any(keyword in label for keyword in ENEMY_KEYWORDS):
                matches.append({"label": obj.get("label"), "confidence": obj.get("confidence"), "bbox": self._normalize_bbox(bbox)})
        return matches[:5]

    def _extract_resources(self, text_zones: Dict[str, Dict[str, object]]):
        import re
        resources = {}
        for zone_name, zone in (text_zones or {}).items():
            text, lowered = str(zone.get("text") or ""), str(zone.get("text") or "").lower()
            for resource, keywords in RESOURCE_KEYWORDS.items():
                if resource in resources or not any(keyword in lowered for keyword in keywords): continue
                match = re.search(r"(\d+)\s*/\s*(\d+)", text)
                if match:
                    resources[resource] = {"current": int(match.group(1)), "max": int(match.group(2)), "zone": zone_name}
                else:
                    numbers = re.findall(r"\d+", text)
                    if len(numbers) >= 2:
                        resources[resource] = {"current": int(numbers[0]), "max": int(numbers[1]), "zone": zone_name}
                    elif len(numbers) == 1 and len(numbers[0]) >= 2 and len(numbers[0]) % 2 == 0:
                        half = len(numbers[0]) // 2
                        resources[resource] = {"current": int(numbers[0][:half]), "max": int(numbers[0][half:]), "zone": zone_name}
        return resources

    def _sanitize_player(self, candidate):
        if not isinstance(candidate, dict): return None
        try:
            bbox = self._normalize_bbox(candidate.get("bbox"))
            if not bbox: return None
            return {"label": candidate.get("label") or "player", "confidence": candidate.get("confidence"), "bbox": bbox, "extra": candidate.get("extra")}
        except Exception: return None

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            topics = [(VISION_MEAN_TOPIC, 0), (VISION_SNAPSHOT_TOPIC, 0), (OCR_TEXT_TOPIC, 0),
                      (OCR_EASY_TOPIC, 0), (SIMPLE_OCR_TOPIC, 0)]
            if OBJECT_TOPIC: topics.append((OBJECT_TOPIC, 0))
            if OBSERVATION_TOPIC: topics.append((OBSERVATION_TOPIC, 0))
            client.subscribe(topics)
            client.publish(SCENE_TOPIC, json.dumps({"ok": True, "event": "scene_agent_ready"}))
        else:
            client.publish(SCENE_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _maybe_publish(self, client):
        now = time.time()
        if now - self.state["snapshot_ts"] > WINDOW_SEC or not self.state["mean"]: return
        fuse_start = time.perf_counter()
        entries = [self.state["easy_text"]] if self.state["easy_text"] else [self.state["simple_text"]] if self.state["simple_text"] else []
        text_payload = [self._normalize_text(entry) for entry in entries] if entries and NORMALIZE_TEXT else entries
        obs = self.state.get("observation", {})
        objects = obs.get("yolo_objects") or self.state.get("objects", [])
        text_zones = obs.get("text_zones") or self.state.get("text_zones", {})
        player_entry = self._sanitize_player(obs.get("player_candidate")) or self.state.get("player_candidate")
        if player_entry and (now - self.state.get("player_candidate_ts", 0)) > WINDOW_SEC * 3: player_entry = None
        if not player_entry: player_entry = self._extract_player(objects)
        if not player_entry and objects: player_entry = {"label": objects[0].get("label") or "player", "confidence": objects[0].get("confidence"), "bbox": self._normalize_bbox(objects[0].get("bbox") or objects[0].get("box"))}
        if not player_entry: player_entry = self.state.get("player_candidate") or {"label": "player_estimate", "confidence": 0.05, "bbox": [0.35, 0.35, 0.65, 0.85]}
        enemies = self._extract_enemies(objects)
        resources = self._extract_resources(text_zones) or self._extract_resources({"aggregate": {"text": "\n".join(text_payload)}})
        payload = {"ok": True, "event": "scene_update", "mean": self.state["mean"][-1], "trend": list(self.state["mean"]),
                   "text": text_payload, "objects": objects, "objects_ts": self.state.get("objects_ts", 0.0),
                   "text_zones": text_zones, "player": player_entry, "enemies": enemies, "resources": resources, "timestamp": now}
        if isinstance(player_entry, dict) and player_entry.get("bbox"):
            bbox = player_entry["bbox"]
            payload["player_center"] = [round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4)]
        targets = []
        for name, zone in (text_zones or {}).items():
            text, bbox = str(zone.get("text") or "").strip(), zone.get("bbox") or zone.get("box")
            if not text or not bbox: continue
            norm_bbox = self._normalize_bbox(bbox)
            if not norm_bbox: continue
            targets.append({"label": text, "zone": name, "bbox": norm_bbox, "center": [round((norm_bbox[0] + norm_bbox[2]) / 2.0, 4), round((norm_bbox[1] + norm_bbox[3]) / 2.0, 4)]})
        if targets: payload["targets"] = targets
        lower_text = " ".join(text_payload).lower()
        if (lower_text and any(kw in lower_text for kw in DEATH_KEYWORDS)) or self._object_matches(objects):
            payload["flags"], payload["death_reason"] = {"death": True}, "text_or_object_match"
        elif text_payload and self._symbolic_text_only(text_payload) and payload.get("mean") <= DEATH_SYMBOL_MEAN_THRESHOLD and len(objects) <= DEATH_OBJECT_THRESHOLD:
            if self._symbolic_candidate_since == 0.0: self._symbolic_candidate_since = now
            if (now - self._symbolic_candidate_since) >= DEATH_SYMBOL_PERSIST: payload["flags"], payload["death_reason"] = {"death": True}, "symbolic_text"
        else: self._symbolic_candidate_since = 0.0
        if payload.get("flags", {}).get("death"): payload["death_text"] = text_payload
        client.publish(SCENE_TOPIC, json.dumps(payload))
        fuse_ms = (time.perf_counter() - fuse_start) * 1000.0
        emit_latency(
            client,
            "fuse",
            fuse_ms,
            sla_ms=SLA_STAGE_FUSE_MS,
            tags={"objects": len(objects), "texts": len(text_payload)},
            agent="scene_agent",
        )

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: data = {"raw": msg.payload}
        if msg.topic == VISION_MEAN_TOPIC:
            if isinstance(data, dict) and data.get("mean") is not None:
                self.state["mean"].append(float(data["mean"]))
                self.state["snapshot_ts"] = time.time()
        elif msg.topic == VISION_SNAPSHOT_TOPIC: self.state["snapshot_ts"] = time.time()
        elif OBJECT_TOPIC and msg.topic == OBJECT_TOPIC:
            if isinstance(data, dict) and isinstance(data.get("objects"), list):
                self.state["objects"], self.state["objects_ts"] = data["objects"], time.time()
        elif OBSERVATION_TOPIC and msg.topic == OBSERVATION_TOPIC:
            if isinstance(data, dict):
                self.state["observation"], self.state["observation_ts"] = data, time.time()
                if isinstance(data.get("yolo_objects"), list): self.state["objects"], self.state["objects_ts"] = data["yolo_objects"], time.time()
                if isinstance(data.get("text_zones"), dict): self.state["text_zones"] = data["text_zones"]
                if self.state["text_zones"]: self.state["easy_text"] = "\n".join([str(z.get("text") or "") for z in self.state["text_zones"].values() if z.get("text")])
                if self._sanitize_player(data.get("player_candidate")): self.state["player_candidate"], self.state["player_candidate_ts"] = self._sanitize_player(data["player_candidate"]), time.time()
        elif msg.topic in (OCR_TEXT_TOPIC, OCR_EASY_TOPIC):
            if isinstance(data, dict):
                self.state["easy_text"] = (str(data.get("text") or "")).strip()
                # Process detailed results with coordinates
                results = data.get("results")
                if isinstance(results, list):
                    new_zones = {}
                    for i, res in enumerate(results):
                        # Use text_index as key or text prefix
                        key = f"ocr_{i}_{res.get('text', '')[:10]}"
                        new_zones[key] = {
                            "text": res.get("text"),
                            "bbox": res.get("box"),
                            "confidence": res.get("conf", 1.0)
                        }
                    self.state["text_zones"] = new_zones
            else:
                self.state["easy_text"] = str(data).strip()
        elif msg.topic == SIMPLE_OCR_TOPIC:
            raw = data.get("text") if isinstance(data, dict) else data
            self.state["simple_text"] = (str(raw) if raw is not None else "").strip()
        self._maybe_publish(client)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()

def _handle_signal(signum, frame):
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = SceneAgent()
    agent.client.connect(MQTT_HOST, MQTT_PORT, 30)
    agent.client.loop_start()
    stop_event.wait()
    agent.client.loop_stop()
    agent.client.disconnect()

if __name__ == "__main__":
    main()
