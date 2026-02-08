#!/usr/bin/env python3
"""
Progress Agent (Meta-Controller).
Monitors learning progress. If stagnation is detected, consults LLM and intervenes.
Exposes a Web Dashboard on port 8000.
"""
import json
import logging
import math
import os
import re
import threading
import time
import requests
from collections import deque
from flask import Flask, jsonify, render_template_string
import paho.mqtt.client as mqtt
from llm_gate import acquire_gate, blocked_reason, release_gate

# --- CONFIG ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC", "vision/objects")
OCR_TOPIC = os.getenv("OCR_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
LLM_ENDPOINT = os.getenv("TEACHER_LOCAL_ENDPOINT", "http://10.0.0.230:11434/v1/chat/completions")
LLM_MODEL = os.getenv("TEACHER_OPENAI_MODEL", "gpt-4o-mini")
LLM_GATE_LOG_SEC = float(os.getenv("LLM_GATE_LOG_SEC", "15"))

STUCK_THRESHOLD_SEC = 300  # 5 minutes without significant reward
MIN_REWARD_THRESHOLD = 0.05 # What counts as "good" reward
PROGRESS_TOPIC = os.getenv("PROGRESS_TOPIC", "progress/status")
UNDERSTANDING_TOPIC = os.getenv("UNDERSTANDING_TOPIC", "progress/understanding")
UNDERSTANDING_PUBLISH_SEC = float(os.getenv("UNDERSTANDING_PUBLISH_SEC", "10"))
UNDERSTANDING_EMBED_SAMPLE_SEC = float(os.getenv("UNDERSTANDING_EMBED_SAMPLE_SEC", "1.0"))
LOCATION_SIM_THRESHOLD = float(os.getenv("UNDERSTANDING_LOCATION_SIM", "0.86"))
LOCATION_EMA_ALPHA = float(os.getenv("UNDERSTANDING_LOCATION_EMA", "0.1"))
LOCATION_MAX = int(os.getenv("UNDERSTANDING_LOCATION_MAX", "200"))
LOCATION_WINDOW = int(os.getenv("UNDERSTANDING_LOCATION_WINDOW", "120"))
LOCATION_VOCAB_MAX = int(os.getenv("UNDERSTANDING_LOCATION_VOCAB_MAX", "500"))
OCR_VOCAB_MAX = int(os.getenv("UNDERSTANDING_OCR_VOCAB_MAX", "800"))
MIN_EMBED_DIM = int(os.getenv("UNDERSTANDING_MIN_EMBED_DIM", "64"))
GROUNDING_RADIUS = float(os.getenv("UNDERSTANDING_GROUNDING_RADIUS", "0.08"))
CURSOR_STALE_SEC = float(os.getenv("UNDERSTANDING_CURSOR_STALE_SEC", "1.0"))
HTTP_PORT = 8000

logging.basicConfig(level=logging.INFO, format="[progress] %(message)s")
logger = logging.getLogger("progress")

# --- STATE ---
state = {
    "last_reward_time": time.time(),
    "total_reward": 0.0,
    "recent_rewards": deque(maxlen=100),
    "status": "OK",
    "last_intervention": "None",
    "scene_desc": "Unknown",
    "scene_ts": 0.0,
    "objects": {"count": 0, "labels": [], "ts": 0.0},
    "ocr": {"text": "", "ts": 0.0},
    "simple_ocr": {"text": "", "ts": 0.0},
    "cursor": {"ok": False, "x_norm": None, "y_norm": None, "ts": 0.0},
    "understanding": {
        "locations": {
            "count": 0,
            "unique": 0,
            "assignments": 0,
            "new_rate": 0.0,
            "revisit_rate": 0.0,
            "current_id": None,
            "current_similarity": 0.0,
            "current_age_sec": -1,
            "transitions": 0,
            "top": [],
        },
        "objects": {
            "vocab_size": 0,
            "new_rate": 0.0,
            "last_new": [],
            "last_count": 0,
        },
        "ocr": {
            "vocab_size": 0,
            "new_rate": 0.0,
            "last_new": [],
            "last_count": 0,
        },
        "embeddings": {
            "delta_last": None,
            "delta_avg": None,
        },
        "grounding": {
            "clicks": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "targeted_clicks": 0,
            "cursor_clicks": 0,
            "clicks_with_targets": 0,
            "clicks_with_objects": 0,
            "hits_on_targets": 0,
            "hits_on_objects": 0,
            "target_hit_rate": 0.0,
            "object_hit_rate": 0.0,
            "last_hit": None,
            "last_reason": "",
        },
    },
}
lock = threading.Lock()
last_gate_log = 0.0

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
OCR_TOKEN_MIN_LEN = int(os.getenv("OCR_TARGET_MIN_LEN", "2"))
OCR_TOKEN_MIN_ALPHA_RATIO = float(os.getenv("OCR_TARGET_MIN_ALPHA_RATIO", "0.6"))
OCR_TOKEN_EXCLUDE_REGEX = os.getenv("OCR_TARGET_EXCLUDE_REGEX", "").strip()
OCR_TOKEN_EXCLUDE = re.compile(OCR_TOKEN_EXCLUDE_REGEX) if OCR_TOKEN_EXCLUDE_REGEX else None

locations = []
location_id_counter = 0
location_window = deque(maxlen=LOCATION_WINDOW)
transition_counts = {}
last_location_id = None
last_location_started = 0.0
last_location_similarity = 0.0
last_embed_processed = 0.0
last_embedding_vec = None
embedding_delta_avg = None
embedding_delta_last = None
object_vocab_global = set()
ocr_vocab_global = set()
object_new_rate_avg = None
ocr_new_rate_avg = None
last_new_objects = []
last_new_tokens = []
last_scene_objects = []
last_scene_targets = []
last_scene_ts = 0.0
last_scene_emb_ts = 0.0
last_cursor = {"ok": False, "x_norm": None, "y_norm": None, "ts": 0.0}
grounding_stats = {
    "clicks": 0,
    "hits": 0,
    "last_hit": None,
    "last_reason": "",
    "targeted_clicks": 0,
    "cursor_clicks": 0,
    "clicks_with_targets": 0,
    "clicks_with_objects": 0,
    "hits_on_targets": 0,
    "hits_on_objects": 0,
}

def _now() -> float:
    return time.time()

def _age(ts: float) -> int:
    if not ts:
        return -1
    return int(max(0, _now() - ts))

def _ema(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return value
    return prev * (1.0 - alpha) + value * alpha

def _normalize_embedding(vec: list) -> list | None:
    if not isinstance(vec, list) or len(vec) < MIN_EMBED_DIM:
        return None
    total = 0.0
    out = []
    for val in vec:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        out.append(fval)
        total += fval * fval
    if total <= 0.0:
        return None
    inv = 1.0 / math.sqrt(total)
    return [val * inv for val in out]

def _dot(a: list, b: list) -> float:
    total = 0.0
    for av, bv in zip(a, b):
        total += av * bv
    return total

def _extract_tokens(texts: list[str]) -> list[str]:
    tokens = []
    for entry in texts:
        if not entry:
            continue
        for token in TOKEN_RE.findall(entry.lower()):
            if len(token) < OCR_TOKEN_MIN_LEN:
                continue
            if OCR_TOKEN_EXCLUDE and OCR_TOKEN_EXCLUDE.search(token):
                continue
            alpha = sum(1 for ch in token if ch.isalpha())
            if (alpha / max(1, len(token))) < OCR_TOKEN_MIN_ALPHA_RATIO:
                continue
            tokens.append(token)
    return tokens

def _extract_labels(objects: list[dict]) -> list[str]:
    labels = []
    for obj in objects or []:
        label = obj.get("label") or obj.get("class") or obj.get("name")
        if label:
            labels.append(str(label).lower())
    return labels

def _extract_bbox(entry: dict) -> list[float] | None:
    bbox = entry.get("bbox") or entry.get("box")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) > 1.5:
        return None
    return [x1, y1, x2, y2]

def _center_from_bbox(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _point_in_bbox(x: float, y: float, bbox: list[float] | None) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def _location_summary(now: float) -> dict:
    top = sorted(locations, key=lambda entry: entry["count"], reverse=True)[:5]
    top_summary = []
    for loc in top:
        top_summary.append({
            "id": loc["id"],
            "visits": loc["count"],
            "last_seen_sec": int(max(0.0, now - loc["last_ts"])),
        })
    window_total = len(location_window)
    new_rate = sum(1 for val in location_window if val) / window_total if window_total else 0.0
    current_age = -1
    if last_location_started:
        current_age = int(max(0.0, now - last_location_started))
    return {
        "count": len(locations),
        "unique": len(locations),
        "assignments": sum(loc["count"] for loc in locations),
        "new_rate": round(new_rate, 3),
        "revisit_rate": round(1.0 - new_rate, 3) if window_total else 0.0,
        "current_id": last_location_id,
        "current_similarity": round(last_location_similarity, 3),
        "current_age_sec": current_age,
        "transitions": sum(transition_counts.values()),
        "top": top_summary,
    }

def _assign_location(vec: list, now: float) -> tuple[dict, bool, float]:
    global location_id_counter, last_location_id, last_location_started, last_location_similarity
    best_loc = None
    best_sim = -1.0
    for loc in locations:
        sim = _dot(loc["vec"], vec)
        if sim > best_sim:
            best_sim = sim
            best_loc = loc
    is_new = False
    prev_location = last_location_id
    if best_loc and best_sim >= LOCATION_SIM_THRESHOLD:
        best_loc["vec"] = _normalize_embedding([
            (1.0 - LOCATION_EMA_ALPHA) * a + LOCATION_EMA_ALPHA * b
            for a, b in zip(best_loc["vec"], vec)
        ]) or best_loc["vec"]
    else:
        location_id_counter += 1
        best_loc = {
            "id": location_id_counter,
            "vec": vec,
            "count": 0,
            "first_ts": now,
            "last_ts": now,
            "object_vocab": set(),
            "ocr_vocab": set(),
            "object_known_avg": None,
            "ocr_known_avg": None,
            "novel_objects": 0,
            "novel_ocr": 0,
        }
        locations.append(best_loc)
        is_new = True
        if len(locations) > LOCATION_MAX:
            oldest = min(locations, key=lambda entry: entry["last_ts"])
            if oldest in locations:
                locations.remove(oldest)
        best_sim = 1.0
    best_loc["count"] += 1
    best_loc["last_ts"] = now
    last_location_similarity = best_sim
    if prev_location is None or best_loc["id"] != prev_location:
        if prev_location is not None:
            key = f"{prev_location}->{best_loc['id']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        last_location_id = best_loc["id"]
        last_location_started = now
    location_window.append(is_new)
    return best_loc, is_new, best_sim

def _update_location_vocab(location: dict, labels: list[str], tokens: list[str]) -> None:
    global object_new_rate_avg, ocr_new_rate_avg, last_new_objects, last_new_tokens
    new_objects = []
    if labels:
        known = 0
        for label in labels:
            if label in location["object_vocab"]:
                known += 1
            elif len(location["object_vocab"]) < LOCATION_VOCAB_MAX:
                location["object_vocab"].add(label)
                new_objects.append(label)
        known_ratio = known / max(1, len(labels))
        location["object_known_avg"] = _ema(location["object_known_avg"], known_ratio, 0.2)
        location["novel_objects"] += len(new_objects)
    new_tokens = []
    if tokens:
        known = 0
        for token in tokens:
            if token in location["ocr_vocab"]:
                known += 1
            elif len(location["ocr_vocab"]) < LOCATION_VOCAB_MAX:
                location["ocr_vocab"].add(token)
                new_tokens.append(token)
        known_ratio = known / max(1, len(tokens))
        location["ocr_known_avg"] = _ema(location["ocr_known_avg"], known_ratio, 0.2)
        location["novel_ocr"] += len(new_tokens)
    if labels:
        new_global = []
        for label in labels:
            if label not in object_vocab_global:
                object_vocab_global.add(label)
                new_global.append(label)
        object_new_rate_avg = _ema(object_new_rate_avg, len(new_global) / max(1, len(labels)), 0.2)
        last_new_objects = new_global[:8]
    if tokens:
        new_global = []
        for token in tokens:
            if token not in ocr_vocab_global:
                if len(ocr_vocab_global) >= OCR_VOCAB_MAX:
                    break
                ocr_vocab_global.add(token)
                new_global.append(token)
        ocr_new_rate_avg = _ema(ocr_new_rate_avg, len(new_global) / max(1, len(tokens)), 0.2)
        last_new_tokens = new_global[:8]

def _update_understanding_state(now: float) -> None:
    summary = _location_summary(now)
    with lock:
        state["understanding"]["locations"] = summary
        state["understanding"]["objects"] = {
            "vocab_size": len(object_vocab_global),
            "new_rate": round(object_new_rate_avg or 0.0, 3),
            "last_new": last_new_objects,
            "last_count": len(_extract_labels(last_scene_objects)),
        }
        state["understanding"]["ocr"] = {
            "vocab_size": len(ocr_vocab_global),
            "new_rate": round(ocr_new_rate_avg or 0.0, 3),
            "last_new": last_new_tokens,
            "last_count": len(_extract_tokens([state.get("scene_desc", "")])),
        }
        state["understanding"]["embeddings"] = {
            "delta_last": round(embedding_delta_last, 4) if embedding_delta_last is not None else None,
            "delta_avg": round(embedding_delta_avg, 4) if embedding_delta_avg is not None else None,
        }
        hit_rate = (grounding_stats["hits"] / grounding_stats["clicks"]) if grounding_stats["clicks"] else 0.0
        target_hit_rate = (grounding_stats["hits_on_targets"] / grounding_stats["clicks_with_targets"]) if grounding_stats["clicks_with_targets"] else 0.0
        object_hit_rate = (grounding_stats["hits_on_objects"] / grounding_stats["clicks_with_objects"]) if grounding_stats["clicks_with_objects"] else 0.0
        state["understanding"]["grounding"] = {
            "clicks": grounding_stats["clicks"],
            "hits": grounding_stats["hits"],
            "hit_rate": round(hit_rate, 3),
            "targeted_clicks": grounding_stats["targeted_clicks"],
            "cursor_clicks": grounding_stats["cursor_clicks"],
            "clicks_with_targets": grounding_stats["clicks_with_targets"],
            "clicks_with_objects": grounding_stats["clicks_with_objects"],
            "hits_on_targets": grounding_stats["hits_on_targets"],
            "hits_on_objects": grounding_stats["hits_on_objects"],
            "target_hit_rate": round(target_hit_rate, 3),
            "object_hit_rate": round(object_hit_rate, 3),
            "last_hit": grounding_stats["last_hit"],
            "last_reason": grounding_stats["last_reason"],
        }

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    topics = [
        (REWARD_TOPIC, 0),
        (SCENE_TOPIC, 0),
        (OBJECT_TOPIC, 0),
        (OCR_TOPIC, 0),
        (SIMPLE_OCR_TOPIC, 0),
        (ACT_CMD_TOPIC, 0),
        (CURSOR_TOPIC, 0),
    ]
    client.subscribe(topics)
    logger.info("Connected to MQTT")

def on_message(client, userdata, msg):
    global state
    try:
        payload = json.loads(msg.payload.decode())
        
        if msg.topic == REWARD_TOPIC:
            r = float(payload.get("reward", 0.0))
            with lock:
                if r > MIN_REWARD_THRESHOLD:
                    state["last_reward_time"] = _now()
                state["total_reward"] += r
                state["recent_rewards"].append(r)
                
        elif msg.topic == SCENE_TOPIC:
            txt = payload.get("text", [])
            objects = payload.get("objects") or []
            targets = payload.get("targets") or []
            embedding = payload.get("embeddings") or payload.get("embedding")
            now = _now()
            with lock:
                state["scene_desc"] = " ".join(txt)[:200]
                state["scene_ts"] = payload.get("timestamp", now)
            if isinstance(objects, list):
                global last_scene_objects, last_scene_targets, last_scene_ts, last_scene_emb_ts
                last_scene_objects = objects
                last_scene_targets = targets if isinstance(targets, list) else []
                last_scene_ts = now
                last_scene_emb_ts = payload.get("embeddings_ts", 0.0) or payload.get("embedding_ts", 0.0) or 0.0
                labels = _extract_labels(objects)
                with lock:
                    state["objects"] = {
                        "count": len(objects),
                        "labels": labels[:8],
                        "ts": payload.get("timestamp", now),
                        "source": "scene",
                    }
            if isinstance(embedding, list):
                global last_embed_processed
                if (now - last_embed_processed) >= UNDERSTANDING_EMBED_SAMPLE_SEC:
                    last_embed_processed = now
                    vec = _normalize_embedding(embedding)
                    if vec is not None:
                        global last_embedding_vec, embedding_delta_avg, embedding_delta_last
                        if last_embedding_vec is not None:
                            delta = 1.0 - _dot(vec, last_embedding_vec)
                            embedding_delta_last = delta
                            embedding_delta_avg = _ema(embedding_delta_avg, delta, 0.2)
                        last_embedding_vec = vec
                        labels = _extract_labels(objects)
                        ocr_texts = []
                        with lock:
                            ocr_texts.append(state.get("scene_desc", ""))
                            if state.get("ocr", {}).get("text"):
                                ocr_texts.append(state["ocr"]["text"])
                            if state.get("simple_ocr", {}).get("text"):
                                ocr_texts.append(state["simple_ocr"]["text"])
                        tokens = _extract_tokens(ocr_texts)
                        loc, is_new, sim = _assign_location(vec, now)
                        _update_location_vocab(loc, labels, tokens)
                        _update_understanding_state(now)
        elif msg.topic == OBJECT_TOPIC:
            objects = payload.get("objects") or []
            labels = []
            for obj in objects:
                lbl = obj.get("label") or obj.get("class") or ""
                if lbl:
                    labels.append(str(lbl))
            with lock:
                state["objects"] = {
                    "count": len(objects),
                    "labels": labels[:8],
                    "ts": payload.get("timestamp", _now()),
                }
        elif msg.topic == OCR_TOPIC:
            txt = payload.get("text") if isinstance(payload, dict) else payload
            with lock:
                state["ocr"] = {"text": str(txt)[:200], "ts": _now()}
        elif msg.topic == SIMPLE_OCR_TOPIC:
            txt = payload.get("text") if isinstance(payload, dict) else payload
            with lock:
                state["simple_ocr"] = {"text": str(txt)[:200], "ts": _now()}
        elif msg.topic == CURSOR_TOPIC:
            ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
            x_norm = payload.get("x_norm") if isinstance(payload, dict) else None
            y_norm = payload.get("y_norm") if isinstance(payload, dict) else None
            if ok and x_norm is not None and y_norm is not None:
                try:
                    last_cursor["ok"] = True
                    last_cursor["x_norm"] = float(x_norm)
                    last_cursor["y_norm"] = float(y_norm)
                    last_cursor["ts"] = _now()
                except (TypeError, ValueError):
                    pass
            with lock:
                state["cursor"] = dict(last_cursor)
        elif msg.topic == ACT_CMD_TOPIC:
            action = payload.get("action") or payload.get("label") or payload.get("act")
            action = str(action).strip().lower()
            if action in ("click_primary", "click_secondary", "click_middle", "mouse_click"):
                click_pos = None
                if isinstance(payload.get("target_norm"), (list, tuple)) and len(payload["target_norm"]) == 2:
                    try:
                        click_pos = (float(payload["target_norm"][0]), float(payload["target_norm"][1]))
                    except (TypeError, ValueError):
                        click_pos = None
                    grounding_stats["targeted_clicks"] += 1
                elif payload.get("x_norm") is not None and payload.get("y_norm") is not None:
                    try:
                        click_pos = (float(payload["x_norm"]), float(payload["y_norm"]))
                    except (TypeError, ValueError):
                        click_pos = None
                elif last_cursor.get("ok") and (_now() - last_cursor.get("ts", 0.0)) <= CURSOR_STALE_SEC:
                    click_pos = (last_cursor.get("x_norm"), last_cursor.get("y_norm"))
                    grounding_stats["cursor_clicks"] += 1
                hit = None
                reason = "no_targets"
                has_targets = bool(last_scene_targets)
                has_objects = bool(last_scene_objects)
                if has_targets:
                    grounding_stats["clicks_with_targets"] += 1
                if has_objects:
                    grounding_stats["clicks_with_objects"] += 1
                if click_pos and (last_scene_targets or last_scene_objects):
                    x, y = click_pos
                    hit = False
                    for entry in (last_scene_targets or []):
                        bbox = _extract_bbox(entry)
                        if bbox and _point_in_bbox(x, y, bbox):
                            hit = True
                            reason = "target_bbox"
                            grounding_stats["hits_on_targets"] += 1
                            break
                    if not hit:
                        for entry in (last_scene_objects or []):
                            bbox = _extract_bbox(entry)
                            if bbox and _point_in_bbox(x, y, bbox):
                                hit = True
                                reason = "object_bbox"
                                grounding_stats["hits_on_objects"] += 1
                                break
                    if not hit:
                        combined = (last_scene_targets or []) + (last_scene_objects or [])
                        best_dist = None
                        best_entry = None
                        for entry in combined:
                            bbox = _extract_bbox(entry)
                            center = entry.get("center") or _center_from_bbox(bbox)
                            if not center:
                                continue
                            dx = x - center[0]
                            dy = y - center[1]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_entry = entry
                        if best_dist is not None and best_dist <= GROUNDING_RADIUS:
                            hit = True
                            if best_entry in (last_scene_targets or []):
                                grounding_stats["hits_on_targets"] += 1
                                reason = "target_center"
                            else:
                                grounding_stats["hits_on_objects"] += 1
                                reason = "object_center"
                        else:
                            reason = "miss"
                grounding_stats["clicks"] += 1
                if hit:
                    grounding_stats["hits"] += 1
                grounding_stats["last_hit"] = hit
                grounding_stats["last_reason"] = reason
                _update_understanding_state(_now())
                
    except Exception as e:
        pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="progress_agent")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# --- LLM INTERVENTION ---
def consult_oracle():
    """Asks LLM what to do when stuck."""
    global last_gate_log
    reason = blocked_reason()
    if reason:
        now = time.time()
        if now - last_gate_log >= LLM_GATE_LOG_SEC:
            logger.info("LLM blocked (%s); skipping intervention", reason)
            last_gate_log = now
        return None
    if not acquire_gate("progress_intervention", wait_s=0.0):
        now = time.time()
        if now - last_gate_log >= LLM_GATE_LOG_SEC:
            logger.info("LLM gate busy; skipping intervention")
            last_gate_log = now
        return None
    try:
        with lock:
            scene = state["scene_desc"]
        
        prompt = f"""
        I am an AI playing a video game. I have been stuck for 5 minutes (no reward).
        Screen text: "{scene}".
        
        Suggest ONE key press to unstick me (e.g., Esc, Space, Enter, I, M).
        Return ONLY the key name.
        """
        
        logger.info("Consulting LLM...")
        # Support for Local LLM (Ollama style) or OpenAI
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False
        }
        
        # Simple request (assuming local endpoint doesn't need auth, or use env var)
        headers = {}
        if "api.openai.com" in LLM_ENDPOINT:
            headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

        resp = requests.post(LLM_ENDPOINT, json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            ans = resp.json()
            # Extract content (handle standard OpenAI format)
            if "choices" in ans:
                suggestion = ans["choices"][0]["message"]["content"].strip().split()[0] # Take first word
                return suggestion
        return None
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None
    finally:
        release_gate()

def perform_intervention(key):
    logger.warning(f"⚠️ INTERVENTION: Pressing '{key}'")
    payload = {
        "action": "keyboard",
        "key": key.lower(),
        "source": "progress_agent_intervention"
    }
    mqtt_client.publish(ACT_CMD_TOPIC, json.dumps(payload))
    with lock:
        state["last_intervention"] = f"Pressed {key} at {time.strftime('%H:%M:%S')}"
        state["last_reward_time"] = time.time() # Reset timer so we don't spam

def monitor_loop():
    while True:
        time.sleep(10)
        now = time.time()
        with lock:
            last = state["last_reward_time"]
            delta = now - last
        
        if delta > STUCK_THRESHOLD_SEC:
            with lock: state["status"] = "STUCK"
            key = consult_oracle()
            if key:
                perform_intervention(key)
        else:
            with lock: state["status"] = "OK"

def publish_loop():
    while True:
        time.sleep(max(1.0, UNDERSTANDING_PUBLISH_SEC))
        payload = {}
        with lock:
            payload = {
                "ok": True,
                "event": "progress_status",
                "timestamp": _now(),
                "status": state.get("status"),
                "total_reward": round(state.get("total_reward", 0.0), 4),
                "last_reward_age": _age(state.get("last_reward_time", 0.0)),
                "scene_age": _age(state.get("scene_ts", 0.0)),
                "understanding": state.get("understanding", {}),
            }
        if PROGRESS_TOPIC:
            mqtt_client.publish(PROGRESS_TOPIC, json.dumps(payload))
        if UNDERSTANDING_TOPIC:
            mqtt_client.publish(UNDERSTANDING_TOPIC, json.dumps({
                "ok": True,
                "event": "understanding_update",
                "timestamp": _now(),
                "understanding": payload.get("understanding", {}),
            }))

# --- LOG READER ---
def read_thought_log(n=5):
    log_path = "/app/logs/thought_process.log"
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-n:]
                for line in reversed(lines):
                    try:
                        entries.append(json.loads(line))
                    except: pass
        except Exception: pass
    return entries

# --- WEB SERVER ---
app = Flask(__name__)

@app.route('/')
def dashboard():
    with lock:
        s = state.copy()
        last_reward_ago = int(time.time() - s["last_reward_time"])
        scene_age = _age(s.get("scene_ts", 0))
        obj_state = s.get("objects", {})
        obj_age = _age(obj_state.get("ts", 0))
        ocr_age = _age(s.get("ocr", {}).get("ts", 0))
        simple_age = _age(s.get("simple_ocr", {}).get("ts", 0))
        understanding = s.get("understanding") or {}
        understanding_view = {
            "locations": understanding.get("locations") or {},
            "objects": understanding.get("objects") or {},
            "ocr": understanding.get("ocr") or {},
            "grounding": understanding.get("grounding") or {},
        }
    
    thoughts = read_thought_log()
    color = "green" if s["status"] == "OK" else "red"
    
    return render_template_string("""
    <html>
    <head>
        <meta refresh="10">
        <style>
            body { background: #111; color: #eee; font-family: monospace; padding: 20px; }
            .box { border: 1px solid #444; padding: 20px; margin: 10px; border-radius: 5px; background: #222; }
            .status { font-size: 24px; font-weight: bold; color: {{ color }}; }
            .thought { margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px; }
            .advice { color: #8f8; font-weight: bold; }
            .scene { color: #aaa; font-size: 0.9em; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
            .label { color: #888; }
        </style>
    </head>
    <body>
        <h1>Progress Manager</h1>
        <div class="grid">
            <div class="box">
                Status: <span class="status">{{ s.status }}</span><br>
                Last Reward: {{ last_reward_ago }}s ago<br>
                Total Reward: {{ s.total_reward | round(2) }}<br>
                Intervention: {{ s.last_intervention }}
            </div>
            <div class="box">
                <div class="label">Scene text (age {{ scene_age }}s)</div>
                <div>{{ s.scene_desc }}</div>
            </div>
            <div class="box">
                <div class="label">Objects (age {{ obj_age }}s)</div>
                <div>Count: {{ obj_state.count or 0 }}</div>
                <div>Labels: {{ obj_state.labels }}</div>
            </div>
            <div class="box">
                <div class="label">OCR Easy (age {{ ocr_age }}s)</div>
                <div>{{ s.ocr.text if s.ocr else "" }}</div>
            </div>
            <div class="box">
                <div class="label">Simple OCR (age {{ simple_age }}s)</div>
                <div>{{ s.simple_ocr.text if s.simple_ocr else "" }}</div>
            </div>
            <div class="box">
                <div class="label">Locations</div>
                <div>Current: {{ understanding.locations.current_id }} (sim {{ understanding.locations.current_similarity }})</div>
                <div>Unique: {{ understanding.locations.unique }}</div>
                <div>New rate: {{ understanding.locations.new_rate }}</div>
                <div>Revisit rate: {{ understanding.locations.revisit_rate }}</div>
            </div>
            <div class="box">
                <div class="label">Object / OCR novelty</div>
                <div>Objects vocab: {{ understanding.objects.vocab_size }} | new rate {{ understanding.objects.new_rate }}</div>
                <div>OCR vocab: {{ understanding.ocr.vocab_size }} | new rate {{ understanding.ocr.new_rate }}</div>
            </div>
            <div class="box">
                <div class="label">Grounding</div>
                <div>Hit rate: {{ understanding.grounding.hit_rate }}</div>
                <div>Clicks: {{ understanding.grounding.clicks }} | Hits: {{ understanding.grounding.hits }}</div>
                <div>Last: {{ understanding.grounding.last_reason }}</div>
            </div>
        </div>
        
        <div class="box">
            <h3>Teacher's Thoughts (Latest)</h3>
            {% for t in thoughts %}
            <div class="thought">
                <div class="advice">💡 {{ t.advice }}</div>
                <div class="scene">👁️ {{ t.scene }}</div>
                <div style="font-size:0.8em; color:#666;">{{ t.timestamp }} | Game: {{ t.game }}</div>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """, s=s, last_reward_ago=last_reward_ago, color=color, thoughts=thoughts,
    scene_age=scene_age, obj_state=obj_state, obj_age=obj_age, ocr_age=ocr_age, simple_age=simple_age,
    understanding=understanding_view)

@app.route('/api/status')
def api_status():
    with lock:
        status = state.copy()
        status["recent_rewards"] = list(state.get("recent_rewards", []))
        status["age"] = {
            "reward": _age(state.get("last_reward_time", 0)),
            "scene": _age(state.get("scene_ts", 0)),
            "objects": _age(state.get("objects", {}).get("ts", 0)),
            "ocr": _age(state.get("ocr", {}).get("ts", 0)),
            "simple_ocr": _age(state.get("simple_ocr", {}).get("ts", 0)),
        }
        return jsonify(status)

def main():
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
    mqtt_client.loop_start()
    
    t = threading.Thread(target=monitor_loop, daemon=True)
    t.start()
    p = threading.Thread(target=publish_loop, daemon=True)
    p.start()
    
    app.run(host='0.0.0.0', port=HTTP_PORT)

if __name__ == "__main__":
    main()
