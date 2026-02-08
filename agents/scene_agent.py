#!/usr/bin/env python3
"""Scene agent that fuses raw perception topics into a high-level scene."""
import difflib
import json
import logging
import os
import re
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

from utils.latency import emit_latency, get_sla_ms
from utils.frame_transport import get_frame_bytes

logger = logging.getLogger("scene_agent")
# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
VISION_MEAN_TOPIC = os.getenv("VISION_MEAN_TOPIC", "vision/mean")
VISION_SNAPSHOT_TOPIC = os.getenv("VISION_SNAPSHOT_TOPIC", "vision/snapshot")
VISION_EMBEDDINGS_TOPIC = os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings")
VISION_PROMPT_TOPIC = os.getenv("VISION_PROMPT_TOPIC", "")
VISION_FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
OBJECT_TOPIC = os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
OBSERVATION_TOPIC = os.getenv("VISION_OBSERVATION_TOPIC", "")
SCENE_FLAGS_TOPIC = os.getenv("SCENE_FLAGS_TOPIC", "scene/flags")
OCR_TEXT_TOPIC = os.getenv("OCR_TEXT_TOPIC", "ocr/text")
OCR_EASY_TOPIC = os.getenv("OCR_EASY_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
WINDOW_SEC = float(os.getenv("SCENE_WINDOW_SEC", "2.0"))
EMBED_PUBLISH_INTERVAL = float(os.getenv("SCENE_EMBED_PUBLISH_INTERVAL", "1.0"))
SCENE_CLASS_PATH = os.getenv("SCENE_CLASS_PATH", "")
YOLO_CLASS_PATH = os.getenv("YOLO_CLASS_PATH", "")
DEEPSTREAM_INPUT_SIZE = int(os.getenv("ENGINE_INPUT_SIZE", "0"))
PROMPT_SCORE_TTL_SEC = float(os.getenv("PROMPT_SCORE_TTL_SEC", "2.5"))
PROMPT_TAG_MIN_SCORE = float(os.getenv("PROMPT_TAG_MIN_SCORE", "0.2"))
PROMPT_TAG_TOP_K = int(os.getenv("PROMPT_TAG_TOP_K", "4"))
PROMPT_TAG_PREFIX = os.getenv("PROMPT_TAG_PREFIX", "prompt_").strip() or "prompt_"
PROMPT_TAG_ALLOW = {label.strip().lower() for label in os.getenv("PROMPT_TAG_ALLOW", "").split(",") if label.strip()}
ENEMY_BAR_ENABLE = os.getenv("SCENE_ENEMY_BAR_ENABLE", "0") != "0"
ENEMY_BAR_FRAME_TOPIC = os.getenv("SCENE_ENEMY_BAR_FRAME_TOPIC", VISION_FRAME_TOPIC)
ENEMY_BAR_INTERVAL = float(os.getenv("SCENE_ENEMY_BAR_INTERVAL", "0.5"))
ENEMY_BAR_MAX_AGE_SEC = float(os.getenv("SCENE_ENEMY_BAR_MAX_AGE_SEC", "1.5"))
ENEMY_BAR_MIN_ASPECT = float(os.getenv("SCENE_ENEMY_BAR_MIN_ASPECT", "3.5"))
ENEMY_BAR_MIN_WIDTH = float(os.getenv("SCENE_ENEMY_BAR_MIN_WIDTH", "0.04"))
ENEMY_BAR_MAX_WIDTH = float(os.getenv("SCENE_ENEMY_BAR_MAX_WIDTH", "0.35"))
ENEMY_BAR_MAX_HEIGHT = float(os.getenv("SCENE_ENEMY_BAR_MAX_HEIGHT", "0.03"))
ENEMY_BAR_MIN_Y = float(os.getenv("SCENE_ENEMY_BAR_MIN_Y", "0.08"))
ENEMY_BAR_MAX_Y = float(os.getenv("SCENE_ENEMY_BAR_MAX_Y", "0.85"))
ENEMY_BAR_MIN_RED_RATIO = float(os.getenv("SCENE_ENEMY_BAR_MIN_RED_RATIO", "0.55"))
ENEMY_BAR_MAX_BARS = int(os.getenv("SCENE_ENEMY_BAR_MAX_BARS", "5"))
ENEMY_BAR_MAX_WIDTH_PX = int(os.getenv("SCENE_ENEMY_BAR_MAX_WIDTH_PX", "720"))
ENEMY_BAR_DEBUG = os.getenv("SCENE_ENEMY_BAR_DEBUG", "0") != "0"
OBJECT_PREFER_OBSERVATION = os.getenv("OBJECT_PREFER_OBSERVATION", "1") != "0"
OBJECT_ALLOW_OBSERVATION_FALLBACK = os.getenv("OBJECT_ALLOW_OBSERVATION_FALLBACK", "1") != "0"
OBJECT_FALLBACK_AFTER_SEC = float(os.getenv("OBJECT_FALLBACK_AFTER_SEC", "1.0"))
NORMALIZE_TEXT = os.getenv("SCENE_NORMALIZE_TEXT", "1") != "0"
OCR_TARGET_MIN_LEN = int(os.getenv("OCR_TARGET_MIN_LEN", "2"))
OCR_TARGET_MIN_CONF = float(os.getenv("OCR_TARGET_MIN_CONF", "0.55"))
OCR_TARGET_MIN_ALPHA_RATIO = float(os.getenv("OCR_TARGET_MIN_ALPHA_RATIO", "0.3"))
OCR_TARGET_EXCLUDE_REGEX = os.getenv("OCR_TARGET_EXCLUDE_REGEX", "").strip()
OCR_TARGET_EXCLUDE = re.compile(OCR_TARGET_EXCLUDE_REGEX) if OCR_TARGET_EXCLUDE_REGEX else None
OCR_TARGET_STABLE_FRAMES = max(1, int(os.getenv("OCR_TARGET_STABLE_FRAMES", "2")))
OCR_TARGET_CONSECUTIVE_MIN = max(1, int(os.getenv("OCR_TARGET_CONSECUTIVE_MIN", "2")))
OCR_AGG_WINDOW = max(1, int(os.getenv("OCR_AGG_WINDOW", "1")))
OCR_AGG_MIN_VOTES = max(1, int(os.getenv("OCR_AGG_MIN_VOTES", "2")))
OCR_AGG_GRID = max(1, int(os.getenv("OCR_AGG_GRID", "12")))
DEATH_KEYWORDS = [kw.strip().lower() for kw in os.getenv("SCENE_DEATH_KEYWORDS", "you have died,resurrect,revive,respawn,resurrect in town,checkpoint").split(",") if kw.strip()]
DEATH_FUZZY_THRESHOLD = float(os.getenv("SCENE_DEATH_FUZZY_THRESHOLD", "0.7"))
DEATH_SKELETON_THRESHOLD = float(os.getenv("SCENE_DEATH_SKELETON_THRESHOLD", "0.72"))
DEATH_SKELETON_MIN_LEN = int(os.getenv("SCENE_DEATH_SKELETON_MIN_LEN", "6"))
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
GENERIC_ENEMY_LABELS = {label.strip().lower() for label in os.getenv("SCENE_GENERIC_ENEMY_LABELS", "person,character,npc").split(",") if label.strip()}
GENERIC_ENEMY_EXCLUDE_RADIUS = float(os.getenv("SCENE_GENERIC_ENEMY_EXCLUDE_RADIUS", "0.08"))
ROLE_HOSTILE_KEYWORDS = {
    label.strip().lower()
    for label in os.getenv("SCENE_ROLE_HOSTILE_KEYWORDS", "enemy,boss,monster,hostile,bandit").split(",")
    if label.strip()
}
ROLE_INTERACT_KEYWORDS = {
    label.strip().lower()
    for label in os.getenv(
        "SCENE_ROLE_INTERACT_KEYWORDS",
        "loot,chest,item,pickup,door,gate,portal,exit,npc,vendor,shop,button,lever",
    ).split(",")
    if label.strip()
}
ROLE_UI_EDGE_MARGIN = float(os.getenv("SCENE_ROLE_UI_EDGE_MARGIN", "0.08"))
ROLE_UI_MIN_OVERLAP = float(os.getenv("SCENE_ROLE_UI_MIN_OVERLAP", "0.3"))
ROLE_MAX_PER_GROUP = max(1, int(os.getenv("SCENE_ROLE_MAX_PER_GROUP", "8")))
SLA_STAGE_FUSE_MS = get_sla_ms("SLA_STAGE_FUSE_MS")

stop_event = threading.Event()

def _load_class_names(paths: List[str]) -> List[str]:
    for raw in paths:
        cleaned = (raw or "").strip()
        if not cleaned:
            continue
        for token in cleaned.split(","):
            path = Path(token.strip())
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    names = [line.strip() for line in handle if line.strip()]
                if names:
                    logger.info("Loaded %d class names from %s", len(names), path)
                    return names
            except Exception as exc:
                logger.warning("Failed to read class names from %s: %s", path, exc)
    return []


CLASS_NAMES = _load_class_names([SCENE_CLASS_PATH, YOLO_CLASS_PATH])


def _target_text_valid(text: str, zone: dict) -> bool:
    trimmed = "".join(ch for ch in text if not ch.isspace())
    if not trimmed or len(trimmed) < OCR_TARGET_MIN_LEN:
        return False
    conf = zone.get("confidence")
    if conf is not None:
        try:
            if float(conf) < OCR_TARGET_MIN_CONF:
                return False
        except (TypeError, ValueError):
            pass
    if OCR_TARGET_EXCLUDE and OCR_TARGET_EXCLUDE.search(trimmed):
        return False
    alpha = sum(1 for ch in trimmed if ch.isalpha())
    if (alpha / max(1, len(trimmed))) < OCR_TARGET_MIN_ALPHA_RATIO:
        return False
    return True


def _normalize_target_text(text: str) -> str:
    if not text:
        return ""
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(normalized.split())


def _bbox_xywh_to_xyxy(bbox: list) -> Optional[list]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(b) for b in bbox)
    except (TypeError, ValueError):
        return None
    return [x, y, x + w, y + h]


def _looks_like_xywh(bbox: list) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    try:
        x, y, w, h = (float(b) for b in bbox)
    except (TypeError, ValueError):
        return False
    if w < 0 or h < 0:
        return False
    if w > x and h > y:
        return False
    return (x + w) <= 1.05 and (y + h) <= 1.05


def _normalize_ocr_bbox(box: list, fmt: Optional[str]) -> Optional[list]:
    if not box or len(box) != 4:
        return None
    fmt = (fmt or "").lower().strip()
    if fmt == "xywh" or (not fmt and _looks_like_xywh(box)):
        return _bbox_xywh_to_xyxy(box)
    return box


def _candidate_chunks(cleaned: str) -> List[str]:
    if not cleaned:
        return []
    chunks = {cleaned}
    splits = cleaned.split()
    if len(splits) >= 2:
        chunks.add(" ".join(splits[:2]))
        chunks.add(" ".join(splits[:3]))
    for token in splits:
        if token:
            chunks.add(token)
    return list(chunks)


def _normalize_skeleton(text: str) -> str:
    """Reduce OCR noise by comparing consonant skeletons with collapsed repeats."""
    filtered = [ch for ch in str(text).lower() if ch.isalnum()]
    if not filtered:
        return ""
    vowels = set("aeiouy")
    skeleton = [ch for ch in filtered if ch not in vowels]
    if not skeleton:
        return ""
    deduped = [skeleton[0]]
    for ch in skeleton[1:]:
        if ch != deduped[-1]:
            deduped.append(ch)
    return "".join(deduped)


def _normalize_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: Optional[float],
    frame_h: Optional[float],
) -> Optional[List[float]]:
    coords = [x1, y1, x2, y2]
    max_val, min_val = max(coords), min(coords)
    if max_val <= 1.5 and min_val >= -0.1:
        nx1, ny1, nx2, ny2 = coords
    else:
        if frame_w and frame_h:
            nx1, ny1, nx2, ny2 = x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h
        elif DEEPSTREAM_INPUT_SIZE > 0:
            size = float(DEEPSTREAM_INPUT_SIZE)
            nx1, ny1, nx2, ny2 = x1 / size, y1 / size, x2 / size, y2 / size
        else:
            return None
    nx1, ny1, nx2, ny2 = [max(0.0, min(1.0, float(v))) for v in (nx1, ny1, nx2, ny2)]
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)]


def _extract_frame_dims(payload: dict) -> tuple[Optional[float], Optional[float]]:
    for key in ("frame_width", "image_width", "width"):
        if key in payload:
            try:
                frame_w = float(payload[key])
                break
            except (TypeError, ValueError):
                frame_w = None
                break
    else:
        frame_w = None
    for key in ("frame_height", "image_height", "height"):
        if key in payload:
            try:
                frame_h = float(payload[key])
                break
            except (TypeError, ValueError):
                frame_h = None
                break
    else:
        frame_h = None
    return frame_w, frame_h


def _parse_ui_boxes(raw: str) -> List[List[float]]:
    boxes: List[List[float]] = []
    if not raw:
        return boxes
    for token in raw.split(";"):
        piece = token.strip()
        if not piece:
            continue
        parts = [segment.strip() for segment in piece.split(",")]
        if len(parts) != 4:
            continue
        try:
            box = [max(0.0, min(1.0, float(value))) for value in parts]
        except (TypeError, ValueError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
    return boxes


ROLE_UI_BOXES = _parse_ui_boxes(
    os.getenv(
        "SCENE_ROLE_UI_BOXES",
        "0.00,0.70,0.34,1.00;0.66,0.70,1.00,1.00;0.72,0.00,1.00,0.36",
    )
)


def _convert_detections_to_objects(payload: dict) -> List[Dict[str, object]]:
    detections = payload.get("detections")
    if not isinstance(detections, list):
        return []
    frame_w, frame_h = _extract_frame_dims(payload)
    objects: List[Dict[str, object]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox") or det.get("box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, w, h = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        x2, y2 = x1 + w, y1 + h
        norm_bbox = _normalize_xyxy(x1, y1, x2, y2, frame_w, frame_h)
        if not norm_bbox:
            continue
        class_id = det.get("class_id")
        label = det.get("label")
        if not label and class_id is not None and CLASS_NAMES:
            try:
                label = CLASS_NAMES[int(class_id)]
            except Exception:
                label = None
        if not label and class_id is not None:
            label = f"class_{class_id}"
        conf = det.get("score")
        if conf is None:
            conf = det.get("confidence")
        if conf is None:
            conf = det.get("conf")
        try:
            conf_val = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_val = 0.0
        objects.append({"label": label or "object", "confidence": round(conf_val, 4), "bbox": norm_bbox})
    return objects

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
            "objects_source": "", "embeddings": [], "embeddings_ts": 0.0, "flags": {}, "flags_ts": 0.0,
            "prompt_scores": {}, "prompt_ts": 0.0,
            "enemy_bars": [], "enemy_bars_ts": 0.0,
        }
        self._last_embed_publish_ts = 0.0
        self._symbolic_candidate_since = 0.0
        self._ocr_token_history = deque(maxlen=OCR_TARGET_STABLE_FRAMES)
        self._ocr_token_consecutive: Dict[str, int] = {}
        self._ocr_zone_history = deque(maxlen=OCR_AGG_WINDOW)

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

    def _update_ocr_stability(self, tokens: List[str]) -> None:
        if OCR_TARGET_STABLE_FRAMES <= 1 and OCR_TARGET_CONSECUTIVE_MIN <= 1:
            return
        normalized = {token for token in tokens if token}
        self._ocr_token_history.append(normalized)
        consecutive = {}
        for token in normalized:
            consecutive[token] = self._ocr_token_consecutive.get(token, 0) + 1
        self._ocr_token_consecutive = consecutive

    def _ocr_target_is_stable(self, token: str) -> bool:
        if OCR_TARGET_STABLE_FRAMES <= 1 and OCR_TARGET_CONSECUTIVE_MIN <= 1:
            return True
        if OCR_TARGET_CONSECUTIVE_MIN > 1:
            if self._ocr_token_consecutive.get(token, 0) >= OCR_TARGET_CONSECUTIVE_MIN:
                return True
        if OCR_TARGET_STABLE_FRAMES > 1 and len(self._ocr_token_history) >= OCR_TARGET_STABLE_FRAMES:
            history = list(self._ocr_token_history)[-OCR_TARGET_STABLE_FRAMES:]
            if all(token in frame for frame in history):
                return True
        return False

    def _update_ocr_history(self, zones: Dict[str, Dict[str, object]]) -> None:
        if OCR_AGG_WINDOW <= 1:
            return
        entries: List[Dict[str, object]] = []
        for zone in (zones or {}).values():
            text = str(zone.get("text") or "").strip()
            if not text:
                continue
            bbox = zone.get("bbox") or zone.get("box")
            norm_bbox = self._normalize_bbox(bbox)
            if not norm_bbox:
                continue
            normalized = _normalize_target_text(text)
            if not normalized:
                continue
            cx = (norm_bbox[0] + norm_bbox[2]) / 2.0
            cy = (norm_bbox[1] + norm_bbox[3]) / 2.0
            entries.append(
                {
                    "text": text,
                    "norm": normalized,
                    "bbox": norm_bbox,
                    "center": (cx, cy),
                    "confidence": zone.get("confidence"),
                }
            )
        self._ocr_zone_history.append(entries)

    def _aggregate_ocr_zones(self) -> tuple[Dict[str, Dict[str, object]], List[str]]:
        if OCR_AGG_WINDOW <= 1 or not self._ocr_zone_history:
            return {}, []
        min_votes = min(OCR_AGG_MIN_VOTES, OCR_AGG_WINDOW)
        buckets: Dict[tuple[int, int], List[Dict[str, object]]] = {}
        for entries in self._ocr_zone_history:
            for entry in entries:
                cx, cy = entry["center"]
                key = (int(cx * OCR_AGG_GRID), int(cy * OCR_AGG_GRID))
                buckets.setdefault(key, []).append(entry)
        aggregated: Dict[str, Dict[str, object]] = {}
        texts: List[str] = []
        idx = 0
        for key, entries in buckets.items():
            counts: Dict[str, int] = {}
            for entry in entries:
                norm = entry.get("norm") or ""
                if not norm:
                    continue
                counts[norm] = counts.get(norm, 0) + 1
            if not counts:
                continue
            best_norm, best_count = max(counts.items(), key=lambda item: item[1])
            if best_count < min_votes:
                continue
            chosen = None
            for frame_entries in reversed(self._ocr_zone_history):
                for entry in frame_entries:
                    if entry.get("norm") != best_norm:
                        continue
                    cx, cy = entry["center"]
                    if (int(cx * OCR_AGG_GRID), int(cy * OCR_AGG_GRID)) == key:
                        chosen = entry
                        break
                if chosen:
                    break
            if chosen is None:
                chosen = entries[-1]
            idx += 1
            zone_key = f"agg_{idx}_{best_norm[:10]}"
            aggregated[zone_key] = {
                "text": chosen.get("text"),
                "bbox": chosen.get("bbox"),
                "confidence": chosen.get("confidence"),
                "votes": best_count,
            }
            if chosen.get("text"):
                texts.append(str(chosen.get("text")))
        return aggregated, texts

    def _prompt_tokens(self, now: float) -> tuple[list[str], Optional[dict]]:
        scores = self.state.get("prompt_scores")
        if not isinstance(scores, dict) or not scores:
            return [], None
        if PROMPT_SCORE_TTL_SEC > 0 and (now - self.state.get("prompt_ts", 0.0)) > PROMPT_SCORE_TTL_SEC:
            return [], None
        items = []
        for label, raw_score in scores.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            normalized = str(label).strip().lower()
            if not normalized:
                continue
            if PROMPT_TAG_ALLOW and normalized not in PROMPT_TAG_ALLOW:
                continue
            items.append((normalized, score))
        if not items:
            return [], None
        items.sort(key=lambda item: item[1], reverse=True)
        tokens = []
        for label, score in items:
            if score < PROMPT_TAG_MIN_SCORE:
                continue
            tokens.append(f"{PROMPT_TAG_PREFIX}{label}")
            if len(tokens) >= PROMPT_TAG_TOP_K:
                break
        return tokens, {label: round(score, 4) for label, score in items}

    def _decode_frame(self, payload: dict) -> Optional["np.ndarray"]:
        if not payload or not isinstance(payload, dict):
            return None
        frame_bytes = get_frame_bytes(payload)
        if not frame_bytes:
            return None
        if isinstance(frame_bytes, memoryview):
            frame_bytes = frame_bytes.tobytes()
        elif isinstance(frame_bytes, bytearray):
            frame_bytes = bytes(frame_bytes)
        if not isinstance(frame_bytes, (bytes, bytearray)):
            if ENEMY_BAR_DEBUG:
                logger.warning("Enemy bar decode skipped: unsupported frame_bytes type=%s", type(frame_bytes))
            return None
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            if ENEMY_BAR_DEBUG:
                logger.warning("Enemy bar detection disabled: cv2/numpy unavailable (%s)", exc)
            return None
        try:
            if len(frame_bytes) < 16:
                return None
            data = np.frombuffer(frame_bytes, dtype=np.uint8)
            if data.size == 0:
                return None
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            if ENEMY_BAR_DEBUG:
                logger.warning(
                    "Enemy bar decode failed: %s (type=%s len=%s)",
                    exc,
                    type(frame_bytes),
                    len(frame_bytes) if hasattr(frame_bytes, "__len__") else "na",
                )
            return None

    def _detect_enemy_bars(self, frame: "np.ndarray") -> List[Dict[str, object]]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return []
        if frame is None or not hasattr(frame, "shape"):
            return []
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return []
        scale = 1.0
        if ENEMY_BAR_MAX_WIDTH_PX > 0 and width > ENEMY_BAR_MAX_WIDTH_PX:
            scale = ENEMY_BAR_MAX_WIDTH_PX / float(width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 70, 70], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 70, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1)
        mask |= cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars: List[Dict[str, object]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue
            aspect = w / float(h)
            w_ratio = w / float(width)
            h_ratio = h / float(height)
            y_center = (y + h / 2.0) / float(height)
            if aspect < ENEMY_BAR_MIN_ASPECT:
                continue
            if w_ratio < ENEMY_BAR_MIN_WIDTH or w_ratio > ENEMY_BAR_MAX_WIDTH:
                continue
            if h_ratio > ENEMY_BAR_MAX_HEIGHT:
                continue
            if y_center < ENEMY_BAR_MIN_Y or y_center > ENEMY_BAR_MAX_Y:
                continue
            roi = mask[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            red_ratio = float(cv2.countNonZero(roi)) / float(roi.size)
            if red_ratio < ENEMY_BAR_MIN_RED_RATIO:
                continue
            x1 = max(0.0, min(1.0, x / float(width)))
            y1 = max(0.0, min(1.0, y / float(height)))
            x2 = max(0.0, min(1.0, (x + w) / float(width)))
            y2 = max(0.0, min(1.0, (y + h) / float(height)))
            bars.append(
                {
                    "label": "enemy_bar",
                    "confidence": round(red_ratio, 3),
                    "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                    "center": [round((x1 + x2) / 2.0, 4), round((y1 + y2) / 2.0, 4)],
                }
            )
        bars.sort(key=lambda b: b.get("confidence", 0.0), reverse=True)
        return bars[: max(1, ENEMY_BAR_MAX_BARS)]

    def _maybe_update_enemy_bars(self, payload: dict) -> None:
        if not ENEMY_BAR_ENABLE:
            return
        now = time.time()
        if ENEMY_BAR_INTERVAL > 0 and (now - self.state.get("enemy_bars_ts", 0.0)) < ENEMY_BAR_INTERVAL:
            return
        try:
            frame = self._decode_frame(payload)
            if frame is None:
                return
            bars = self._detect_enemy_bars(frame)
            if ENEMY_BAR_DEBUG and bars:
                logger.info("Enemy bars detected: %s", bars)
            self.state["enemy_bars"] = bars
            self.state["enemy_bars_ts"] = now
        except Exception as exc:
            if ENEMY_BAR_DEBUG:
                logger.warning("Enemy bar update failed: %s", exc)

    def _normalize_text(self, entry: str) -> str:
        return entry if isinstance(entry, str) else entry

    def _object_matches(self, objects):
        for obj in objects:
            label = str(obj.get("label") or obj.get("text") or obj.get("class") or "").lower()
            if label and any(kw in label for kw in DEATH_KEYWORDS):
                return True
        return False

    def _text_has_death_keyword(self, text_payload: List[str]) -> bool:
        if not text_payload:
            return False
        normalized_entries = []
        for entry in text_payload:
            cleaned = _normalize_target_text(str(entry))
            if cleaned:
                normalized_entries.append(cleaned)
        if not normalized_entries:
            return False
        lower_text = " ".join(normalized_entries)
        if lower_text and any(kw in lower_text for kw in DEATH_KEYWORDS):
            return True
        if DEATH_FUZZY_THRESHOLD <= 0 and DEATH_SKELETON_THRESHOLD <= 0:
            return False
        for cleaned in normalized_entries:
            for candidate in _candidate_chunks(cleaned):
                for keyword in DEATH_KEYWORDS:
                    if not keyword:
                        continue
                    ratio = difflib.SequenceMatcher(None, keyword, candidate).ratio()
                    if ratio >= DEATH_FUZZY_THRESHOLD:
                        return True
                    if DEATH_SKELETON_THRESHOLD > 0:
                        keyword_skeleton = _normalize_skeleton(keyword)
                        candidate_skeleton = _normalize_skeleton(candidate)
                        if (
                            len(keyword_skeleton) >= DEATH_SKELETON_MIN_LEN
                            and len(candidate_skeleton) >= DEATH_SKELETON_MIN_LEN
                        ):
                            skeleton_ratio = difflib.SequenceMatcher(
                                None,
                                keyword_skeleton,
                                candidate_skeleton,
                            ).ratio()
                            if skeleton_ratio >= DEATH_SKELETON_THRESHOLD:
                                return True
        return False

    def _normalize_bbox(self, bbox):
        if not bbox or len(bbox) != 4: return None
        return [round(float(c), 4) for c in bbox]

    def _center_from_bbox(self, bbox: Optional[List[float]]) -> Optional[Tuple[float, float]]:
        if not bbox or len(bbox) != 4:
            return None
        try:
            return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
        except (TypeError, ValueError):
            return None

    def _bbox_intersection_ratio(self, bbox: Optional[List[float]], region: Optional[List[float]]) -> float:
        if not bbox or not region or len(bbox) != 4 or len(region) != 4:
            return 0.0
        try:
            x1 = max(float(bbox[0]), float(region[0]))
            y1 = max(float(bbox[1]), float(region[1]))
            x2 = min(float(bbox[2]), float(region[2]))
            y2 = min(float(bbox[3]), float(region[3]))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            area = max(1e-6, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
            return inter / area
        except (TypeError, ValueError):
            return 0.0

    def _bbox_is_ui(self, bbox: Optional[List[float]]) -> bool:
        if not bbox or len(bbox) != 4:
            return False
        center = self._center_from_bbox(bbox)
        if center:
            cx, cy = center
            if cy >= (1.0 - ROLE_UI_EDGE_MARGIN) and (
                cx <= (ROLE_UI_EDGE_MARGIN * 3.0) or cx >= (1.0 - ROLE_UI_EDGE_MARGIN * 3.0)
            ):
                return True
            if cy <= ROLE_UI_EDGE_MARGIN and cx >= (1.0 - ROLE_UI_EDGE_MARGIN * 3.0):
                return True
        for region in ROLE_UI_BOXES:
            if self._bbox_intersection_ratio(bbox, region) >= ROLE_UI_MIN_OVERLAP:
                return True
        return False

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

    def _extract_enemies(self, objects, player_entry=None):
        matches = []
        player_center = None
        if isinstance(player_entry, dict) and player_entry.get("bbox"):
            try:
                pb = player_entry["bbox"]
                player_center = ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)
            except Exception:
                player_center = None
        for obj in objects:
            bbox = obj.get("bbox") or obj.get("box")
            if not bbox: continue
            norm_bbox = self._normalize_bbox(bbox)
            if not norm_bbox:
                continue
            if self._bbox_is_ui(norm_bbox):
                continue
            raw_label = obj.get("label") or obj.get("class") or obj.get("name")
            label = str(raw_label or "").lower()
            if not ENEMY_KEYWORDS or any(keyword in label for keyword in ENEMY_KEYWORDS):
                matches.append({"label": raw_label, "confidence": obj.get("confidence"), "bbox": norm_bbox})
                continue
            if ALLOW_GENERIC_ENEMIES and label in GENERIC_ENEMY_LABELS:
                if player_center:
                    try:
                        cx, cy = (norm_bbox[0] + norm_bbox[2]) / 2.0, (norm_bbox[1] + norm_bbox[3]) / 2.0
                        dist = (cx - player_center[0]) ** 2 + (cy - player_center[1]) ** 2
                        if dist <= (GENERIC_ENEMY_EXCLUDE_RADIUS ** 2):
                            continue
                    except Exception:
                        pass
                matches.append({"label": raw_label or "enemy", "confidence": obj.get("confidence"), "bbox": norm_bbox})
        return matches[:5]

    def _extract_roles(
        self,
        objects: List[Dict[str, object]],
        player_entry: Optional[Dict[str, object]],
        enemies: List[Dict[str, object]],
        resources: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        roles: Dict[str, object] = {
            "avatar": None,
            "hostiles": [],
            "interactables": [],
            "ui_elements": [],
            "world_objects": [],
            "resource_signals": [],
            "stats": {},
        }
        player_bbox = self._normalize_bbox((player_entry or {}).get("bbox")) if isinstance(player_entry, dict) else None
        enemy_keys = set()
        for enemy in enemies or []:
            bbox = self._normalize_bbox((enemy or {}).get("bbox"))
            if bbox:
                enemy_keys.add(tuple(round(v, 3) for v in bbox))
        for obj in objects or []:
            bbox = self._normalize_bbox(obj.get("bbox") or obj.get("box"))
            if not bbox:
                continue
            center = self._center_from_bbox(bbox)
            if not center:
                continue
            raw_label = obj.get("label") or obj.get("class") or obj.get("name") or "object"
            label = str(raw_label).lower()
            entry = {
                "label": raw_label,
                "confidence": obj.get("confidence"),
                "bbox": bbox,
                "center": [round(center[0], 4), round(center[1], 4)],
            }
            bbox_key = tuple(round(v, 3) for v in bbox)
            is_avatar = bool(player_bbox and self._bbox_intersection_ratio(bbox, player_bbox) >= 0.5)
            is_ui = self._bbox_is_ui(bbox)
            is_hostile = bbox_key in enemy_keys or any(keyword in label for keyword in ROLE_HOSTILE_KEYWORDS)
            is_interactable = any(keyword in label for keyword in ROLE_INTERACT_KEYWORDS)
            if is_avatar and roles["avatar"] is None:
                avatar_entry = dict(entry)
                avatar_entry["role"] = "avatar"
                roles["avatar"] = avatar_entry
            if is_ui:
                if len(roles["ui_elements"]) < ROLE_MAX_PER_GROUP:
                    ui_entry = dict(entry)
                    ui_entry["role"] = "ui_element"
                    roles["ui_elements"].append(ui_entry)
                continue
            if is_hostile and not is_avatar:
                if len(roles["hostiles"]) < ROLE_MAX_PER_GROUP:
                    hostile_entry = dict(entry)
                    hostile_entry["role"] = "hostile"
                    roles["hostiles"].append(hostile_entry)
                continue
            if is_interactable:
                if len(roles["interactables"]) < ROLE_MAX_PER_GROUP:
                    interact_entry = dict(entry)
                    interact_entry["role"] = "interactable"
                    roles["interactables"].append(interact_entry)
                continue
            if len(roles["world_objects"]) < ROLE_MAX_PER_GROUP:
                world_entry = dict(entry)
                world_entry["role"] = "world_object"
                roles["world_objects"].append(world_entry)
        for enemy in enemies or []:
            bbox = self._normalize_bbox((enemy or {}).get("bbox"))
            center = self._center_from_bbox(bbox)
            if not bbox or not center or self._bbox_is_ui(bbox):
                continue
            bbox_key = tuple(round(v, 3) for v in bbox)
            if any(tuple(round(v, 3) for v in h.get("bbox", [])) == bbox_key for h in roles["hostiles"]):
                continue
            if len(roles["hostiles"]) >= ROLE_MAX_PER_GROUP:
                break
            roles["hostiles"].append(
                {
                    "label": enemy.get("label") or "hostile",
                    "confidence": enemy.get("confidence"),
                    "bbox": bbox,
                    "center": [round(center[0], 4), round(center[1], 4)],
                    "role": "hostile",
                }
            )
        for name, info in (resources or {}).items():
            if not isinstance(info, dict):
                continue
            if len(roles["resource_signals"]) >= ROLE_MAX_PER_GROUP:
                break
            roles["resource_signals"].append(
                {
                    "name": str(name),
                    "current": info.get("current"),
                    "max": info.get("max"),
                    "zone": info.get("zone"),
                    "is_plain_value": bool(info.get("is_plain_value")),
                }
            )
        roles["stats"] = {
            "hostile_count": len(roles["hostiles"]),
            "interactable_count": len(roles["interactables"]),
            "ui_count": len(roles["ui_elements"]),
            "world_count": len(roles["world_objects"]),
            "resource_count": len(roles["resource_signals"]),
            "object_count": len(objects or []),
        }
        return roles

    def _extract_resources(self, text_zones: Dict[str, Dict[str, object]]):
        import re
        resources = {}
        idx = 0
        # Common non-game words found in UI/system text
        blacklist = {
            "january", "february", "march", "april", "may", "june", 
            "july", "august", "september", "october", "november", "december",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "today", "yesterday", "now", "time", "version", "system"
        }
        
        for zone_name, zone in (text_zones or {}).items():
            text = str(zone.get("text") or "").strip()
            if not text:
                continue
            
            # 1. Pattern X / Y (e.g. 100/100)
            match_slash = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if match_slash:
                name = self._guess_stat_name(text, f"stat_{idx}")
                if name.lower() not in blacklist:
                    resources[name] = {"current": int(match_slash.group(1)), "max": int(match_slash.group(2)), "zone": zone_name}
                    idx += 1
                continue

            # 2. Pattern X% (e.g. 80%)
            match_pct = re.search(r"(\d+)\s*%", text)
            if match_pct:
                val = int(match_pct.group(1))
                name = self._guess_stat_name(text, f"pct_{idx}")
                if name.lower() not in blacklist:
                    resources[name] = {"current": val, "max": 100, "zone": zone_name}
                    idx += 1
                continue

            # 3. Pattern Label: Value or Value Label (e.g. Score: 123, 30 Ammo)
            match_num = re.search(r"(?::\s*|\s+|^)(\d+)(?:\s+|$)", text)
            if match_num:
                val = int(match_num.group(1))
                # Skip numbers that look like years (1900-2100)
                if 1900 <= val <= 2100:
                    continue
                    
                name = self._guess_stat_name(text, f"val_{idx}")
                if name.lower() not in blacklist:
                    # For plain numbers, we treat them as current values with an unknown or inferred max
                    resources[name] = {"current": val, "max": val if val > 0 else 1, "zone": zone_name, "is_plain_value": True}
                    idx += 1
                continue
        
        return resources

    def _guess_stat_name(self, text: str, default: str) -> str:
        lowered = text.lower()
        for res_type, keywords in RESOURCE_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return res_type
        # Extract the first word if it looks like a label (3+ characters)
        words = re.findall(r"([a-zA-Z]{3,})", lowered)
        if words:
            # Avoid generic and system words
            noise = {
                "the", "and", "you", "have", "with", "from", "for", "not", "this", "that",
                "been", "were", "are", "was", "will", "would", "could", "should", "selected",
                "event", "event_selected", "menu", "options", "settings", "click", "press"
            }
            for word in words:
                if word not in noise:
                    return word
        return default

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
            if VISION_EMBEDDINGS_TOPIC: topics.append((VISION_EMBEDDINGS_TOPIC, 0))
            if VISION_PROMPT_TOPIC: topics.append((VISION_PROMPT_TOPIC, 0))
            if SCENE_FLAGS_TOPIC: topics.append((SCENE_FLAGS_TOPIC, 0))
            if ENEMY_BAR_ENABLE and ENEMY_BAR_FRAME_TOPIC:
                topics.append((ENEMY_BAR_FRAME_TOPIC, 0))
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
        prompt_tokens, prompt_scores = self._prompt_tokens(now)
        flags = dict(self.state.get("flags") or {})
        include_prompt_tokens = flags.get("in_game") is True
        if prompt_tokens and include_prompt_tokens:
            text_payload = list(text_payload) if text_payload else []
            for token in prompt_tokens:
                if token not in text_payload:
                    text_payload.append(token)
        obs = self.state.get("observation", {})
        objects = obs.get("yolo_objects") or self.state.get("objects", [])
        raw_text_zones = obs.get("text_zones") or self.state.get("text_zones", {})
        text_zones = raw_text_zones
        agg_texts: List[str] = []
        if OCR_AGG_WINDOW > 1:
            agg_zones, agg_texts = self._aggregate_ocr_zones()
            if agg_zones:
                text_zones = agg_zones
        if agg_texts:
            if text_payload is None:
                text_payload = []
            for entry in agg_texts:
                normalized = self._normalize_text(entry) if NORMALIZE_TEXT else entry
                if normalized and normalized not in text_payload:
                    text_payload.append(normalized)
        player_entry = self._sanitize_player(obs.get("player_candidate")) or self.state.get("player_candidate")
        if player_entry and (now - self.state.get("player_candidate_ts", 0)) > WINDOW_SEC * 3: player_entry = None
        if not player_entry: player_entry = self._extract_player(objects)
        if not player_entry and objects: player_entry = {"label": objects[0].get("label") or "player", "confidence": objects[0].get("confidence"), "bbox": self._normalize_bbox(objects[0].get("bbox") or objects[0].get("box"))}
        if not player_entry: player_entry = self.state.get("player_candidate") or {"label": "player_estimate", "confidence": 0.05, "bbox": [0.35, 0.35, 0.65, 0.85]}
        enemies = self._extract_enemies(objects, player_entry)
        enemy_bars = []
        if ENEMY_BAR_ENABLE:
            bars_age = now - self.state.get("enemy_bars_ts", 0.0)
            if bars_age <= ENEMY_BAR_MAX_AGE_SEC:
                enemy_bars = list(self.state.get("enemy_bars") or [])
        if enemy_bars:
            for bar in enemy_bars:
                enemies.append(bar)
        resources = self._extract_resources(text_zones) or self._extract_resources({"aggregate": {"text": "\n".join(text_payload)}})
        roles = self._extract_roles(objects, player_entry, enemies, resources)
        role_stats = roles.get("stats") if isinstance(roles, dict) else {}
        stats = {}
        obs_stats = obs.get("stats")
        if isinstance(obs_stats, dict):
            stats.update(obs_stats)
        if isinstance(role_stats, dict):
            stats.setdefault("enemy_count", role_stats.get("hostile_count", len(enemies)))
            stats.setdefault("loot_count", role_stats.get("interactable_count", 0))
            stats.setdefault("ui_count", role_stats.get("ui_count", 0))
            stats.setdefault("world_count", role_stats.get("world_count", 0))
        stats.setdefault("object_count", len(objects))
        stats.setdefault("text_count", len(text_payload))
        payload = {"ok": True, "event": "scene_update", "mean": self.state["mean"][-1], "trend": list(self.state["mean"]),
                   "text": text_payload, "objects": objects, "objects_ts": self.state.get("objects_ts", 0.0),
                   "objects_source": self.state.get("objects_source", ""),
                   "text_zones": text_zones, "player": player_entry, "enemies": enemies, "resources": resources, "timestamp": now,
                   "embeddings": self.state.get("embeddings", []), "embeddings_ts": self.state.get("embeddings_ts", 0.0),
                   "roles": roles, "stats": stats}
        if enemy_bars:
            payload["enemy_bars"] = enemy_bars
        if text_zones is not raw_text_zones:
            payload["text_zones_raw"] = raw_text_zones
            payload["text_zones_agg"] = text_zones
        if prompt_scores:
            payload["prompt_scores"] = prompt_scores
        if flags:
            payload["flags"] = dict(flags)
        if isinstance(player_entry, dict) and player_entry.get("bbox"):
            bbox = player_entry["bbox"]
            payload["player_center"] = [round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4)]
        targets = []
        seen_targets = set()
        for name, zone in (text_zones or {}).items():
            text, bbox = str(zone.get("text") or "").strip(), zone.get("bbox") or zone.get("box")
            if not text or not bbox: continue
            normalized = _normalize_target_text(text)
            if not normalized or normalized in seen_targets:
                continue
            if not _target_text_valid(text, zone):
                continue
            if not self._ocr_target_is_stable(normalized):
                continue
            norm_bbox = self._normalize_bbox(bbox)
            if not norm_bbox: continue
            targets.append({"label": text, "zone": name, "bbox": norm_bbox, "center": [round((norm_bbox[0] + norm_bbox[2]) / 2.0, 4), round((norm_bbox[1] + norm_bbox[3]) / 2.0, 4)]})
            seen_targets.add(normalized)
        if targets: payload["targets"] = targets
        if self._text_has_death_keyword(text_payload) or self._object_matches(objects):
            flags["death"] = True
            payload["flags"], payload["death_reason"] = flags, "text_or_object_match"
        elif text_payload and self._symbolic_text_only(text_payload) and payload.get("mean") <= DEATH_SYMBOL_MEAN_THRESHOLD and len(objects) <= DEATH_OBJECT_THRESHOLD:
            if self._symbolic_candidate_since == 0.0: self._symbolic_candidate_since = now
            if (now - self._symbolic_candidate_since) >= DEATH_SYMBOL_PERSIST:
                flags["death"] = True
                payload["flags"], payload["death_reason"] = flags, "symbolic_text"
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
        publish_now = True
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: data = {"raw": msg.payload}
        if msg.topic == VISION_MEAN_TOPIC:
            if isinstance(data, dict) and data.get("mean") is not None:
                self.state["mean"].append(float(data["mean"]))
                self.state["snapshot_ts"] = time.time()
        elif msg.topic == VISION_SNAPSHOT_TOPIC: self.state["snapshot_ts"] = time.time()
        elif ENEMY_BAR_ENABLE and ENEMY_BAR_FRAME_TOPIC and msg.topic == ENEMY_BAR_FRAME_TOPIC:
            if isinstance(data, dict):
                self._maybe_update_enemy_bars(data)
        elif OBSERVATION_TOPIC and msg.topic == OBSERVATION_TOPIC:
            if isinstance(data, dict):
                now = time.time()
                if not isinstance(data.get("yolo_objects"), list):
                    converted = _convert_detections_to_objects(data)
                    if converted:
                        data["yolo_objects"] = converted
                self.state["observation"], self.state["observation_ts"] = data, now
                if isinstance(data.get("yolo_objects"), list):
                    should_use_obs = OBJECT_PREFER_OBSERVATION
                    if not should_use_obs:
                        last_src = self.state.get("objects_source")
                        last_ts = self.state.get("objects_ts", 0.0)
                        if (
                            OBJECT_ALLOW_OBSERVATION_FALLBACK
                            and OBJECT_FALLBACK_AFTER_SEC > 0
                            and (last_src != "objects_topic" or (now - last_ts) > OBJECT_FALLBACK_AFTER_SEC)
                        ):
                            should_use_obs = True
                    if should_use_obs:
                        self.state["objects"], self.state["objects_ts"] = data["yolo_objects"], now
                        self.state["objects_source"] = "observation"
                if isinstance(data.get("text_zones"), dict) and data.get("text_zones"):
                    self.state["text_zones"] = data["text_zones"]
                    tokens = []
                    for zone in data["text_zones"].values():
                        text = str(zone.get("text") or "").strip()
                        normalized = _normalize_target_text(text)
                        if normalized:
                            tokens.append(normalized)
                    self._update_ocr_stability(tokens)
                if self.state["text_zones"]: self.state["easy_text"] = "\n".join([str(z.get("text") or "") for z in self.state["text_zones"].values() if z.get("text")])
                if self._sanitize_player(data.get("player_candidate")): self.state["player_candidate"], self.state["player_candidate_ts"] = self._sanitize_player(data["player_candidate"]), time.time()
        elif OBJECT_TOPIC and msg.topic == OBJECT_TOPIC and msg.topic != OBSERVATION_TOPIC:
            if isinstance(data, dict):
                if OBJECT_PREFER_OBSERVATION and self.state.get("observation_ts"):
                    obs_age = time.time() - self.state.get("observation_ts", 0.0)
                    obs_objects = (self.state.get("observation") or {}).get("yolo_objects") or []
                    if obs_age <= OBJECT_FALLBACK_AFTER_SEC and obs_objects:
                        publish_now = False
                        return
                converted = _convert_detections_to_objects(data)
                if converted:
                    self.state["objects"], self.state["objects_ts"] = converted, time.time()
                    self.state["objects_source"] = "objects_topic"
                elif isinstance(data.get("objects"), list):
                    self.state["objects"], self.state["objects_ts"] = data["objects"], time.time()
                    self.state["objects_source"] = "objects_topic"
        elif SCENE_FLAGS_TOPIC and msg.topic == SCENE_FLAGS_TOPIC:
            if isinstance(data, dict):
                flags = data.get("flags") if isinstance(data.get("flags"), dict) else data
                if isinstance(flags, dict):
                    merged = dict(self.state.get("flags") or {})
                    merged.update(flags)
                    self.state["flags"] = merged
                    self.state["flags_ts"] = time.time()
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
                        bbox = res.get("box")
                        fmt = res.get("box_format") or res.get("bbox_format") or res.get("format")
                        if bbox:
                            bbox = _normalize_ocr_bbox(bbox, fmt)
                        new_zones[key] = {
                            "text": res.get("text"),
                            "bbox": bbox,
                            "confidence": res.get("conf", 1.0)
                        }
                    self.state["text_zones"] = new_zones
                    tokens = []
                    for zone in new_zones.values():
                        text = str(zone.get("text") or "").strip()
                        normalized = _normalize_target_text(text)
                        if normalized:
                            tokens.append(normalized)
                    self._update_ocr_stability(tokens)
                    self._update_ocr_history(new_zones)
            else:
                self.state["easy_text"] = str(data).strip()
        elif msg.topic == SIMPLE_OCR_TOPIC:
            if isinstance(data, dict):
                items = data.get("items") or []
                texts = []
                zones = {}
                for idx, item in enumerate(items):
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    texts.append(text)
                    box = item.get("box")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        box = _normalize_ocr_bbox(list(box), "xywh")
                        key = f"simple_{idx}_{text[:10]}"
                        zones[key] = {"text": text, "bbox": box, "confidence": item.get("conf")}
                if texts:
                    self.state["simple_text"] = "\n".join(texts)
                    if zones:
                        merged = dict(self.state.get("text_zones") or {})
                        merged.update(zones)
                        self.state["text_zones"] = merged
                        tokens = []
                        for zone in zones.values():
                            normalized = _normalize_target_text(str(zone.get("text") or ""))
                            if normalized:
                                tokens.append(normalized)
                        self._update_ocr_stability(tokens)
                        self._update_ocr_history(zones)
                else:
                    self.state["simple_text"] = ""
            else:
                raw = data
                self.state["simple_text"] = (str(raw) if raw is not None else "").strip()
        elif VISION_PROMPT_TOPIC and msg.topic == VISION_PROMPT_TOPIC:
            if isinstance(data, dict) and isinstance(data.get("scores"), dict):
                self.state["prompt_scores"] = data["scores"]
                self.state["prompt_ts"] = float(data.get("timestamp") or time.time())
        elif msg.topic == VISION_EMBEDDINGS_TOPIC:
            publish_now = False
            if isinstance(data, dict):
                embedding = data.get("embedding") or data.get("embeddings")
                if isinstance(embedding, list):
                    self.state["embeddings"] = embedding
                    self.state["embeddings_ts"] = float(data.get("frame_ts") or data.get("timestamp") or time.time())
                    now = time.time()
                    if now - self._last_embed_publish_ts >= EMBED_PUBLISH_INTERVAL:
                        self._last_embed_publish_ts = now
                        publish_now = True
                else:
                    logger.debug("Embeddings payload ignored: expected list, got %s", type(embedding).__name__)
            else:
                logger.debug("Embeddings payload ignored: expected dict, got %s", type(data).__name__)
        if publish_now:
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
