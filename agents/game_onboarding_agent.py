#!/usr/bin/env python3
"""Autonomous onboarding agent that maps a game's UI and controls."""
from __future__ import annotations

import json
import os
import logging
import threading
import time
import signal
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from control_profile import load_profile, safe_profile, upsert_profile
from llm_client import fetch_control_profile, guess_game_id
from utils.frame_transport import get_frame_bytes

logging.basicConfig(level=os.getenv("ONBOARD_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("game_onboarding")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
SCHEMA_TOPIC = os.getenv("GAME_SCHEMA_TOPIC", "game/schema")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")

PHASE_A_SECONDS = float(os.getenv("ONBOARD_PHASE_A_SEC", "6.0"))
CONTROL_SETTLE_SEC = float(os.getenv("ONBOARD_CONTROL_SETTLE", "1.0"))
POST_PROBE_SEC = float(os.getenv("ONBOARD_POST_PROBE", "1.0"))
FRAME_DOWNSCALE = int(os.getenv("ONBOARD_FRAME_DOWNSCALE", "240"))
VARIANCE_THRESHOLD = float(os.getenv("ONBOARD_VARIANCE_THRESHOLD", "35.0"))
HUD_AREA_MIN = float(os.getenv("ONBOARD_HUD_AREA_MIN", "0.01"))
HUD_AREA_MAX = float(os.getenv("ONBOARD_HUD_AREA_MAX", "0.2"))
CONTROL_DIFF_THRESHOLD = float(os.getenv("ONBOARD_CONTROL_DIFF", "8.0"))
EXTENDED_DIFF_MIN = float(os.getenv("ONBOARD_EXTENDED_DIFF_MIN", str(CONTROL_DIFF_THRESHOLD)))

CONTROL_PROBES = [
    ("KEY_W", {"action": "key_press", "key": "w"}, "move_forward"),
    ("KEY_S", {"action": "key_press", "key": "s"}, "move_backward"),
    ("KEY_A", {"action": "key_press", "key": "a"}, "move_left"),
    ("KEY_D", {"action": "key_press", "key": "d"}, "move_right"),
    ("KEY_Q", {"action": "key_press", "key": "q"}, "skill_q"),
    ("KEY_E", {"action": "key_press", "key": "e"}, "skill_e"),
    ("KEY_R", {"action": "key_press", "key": "r"}, "skill_r"),
    ("KEY_F", {"action": "key_press", "key": "f"}, "interact"),
    ("MOUSE_LEFT", {"action": "click_primary"}, "basic_attack"),
    ("MOUSE_RIGHT", {"action": "click_secondary"}, "alt_attack"),
]

# Default skill/flask keys for click-to-move ARPGs.
ARPG_DEFAULT_KEYS = ["q", "w", "e", "r", "t", "1", "2", "3", "4", "5", "space"]

# Safe probes used only when LLM/profile is low-confidence: avoid risky/system keys.
SAFE_PROBES = [
    ("MOUSE_MOVE", {"action": "mouse_move", "dx": 0, "dy": 0}, "mouse_move"),
    ("MOUSE_LEFT", {"action": "click_primary"}, "basic_attack"),
]

stop_event = threading.Event()
PROBE_WITHOUT_PROFILE = os.getenv("ONBOARD_PROBE_WITHOUT_PROFILE", "0") != "0"
REQUEST_LLM_PROFILE = os.getenv("ONBOARD_REQUEST_LLM_PROFILE", "1") != "0"
REQUEST_LLM_GAME_ID = os.getenv("ONBOARD_REQUEST_LLM_GAME_ID", "1") != "0"
LLM_CONF_MIN_KEYS = int(os.getenv("LLM_CONF_MIN_KEYS", "1"))
DISABLE_PROBES = os.getenv("ONBOARD_DISABLE_PROBES", "1") == "1"
SAFE_KEY_BLACKLIST = {k.strip().lower() for k in os.getenv("ONBOARD_SAFE_KEY_BLACKLIST", "m,tab,i,esc").split(",") if k.strip()}
FORCE_MOUSE_MOVE = os.getenv("ONBOARD_FORCE_MOUSE_MOVE", "1") != "0"
PREFER_LOCAL_PROFILE = os.getenv("ONBOARD_PREFER_LOCAL_PROFILE", "1") != "0"
EXTENDED_PROBE = os.getenv("ONBOARD_EXTENDED_PROBE", "1") != "0"
GAME_ID_OVERRIDE = os.getenv("ONBOARD_GAME_ID_OVERRIDE", "").strip()
GAME_IDENTITY_TOPIC = os.getenv("GAME_IDENTITY_TOPIC", "game/identity")

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

def _normalize_game_id(value: str) -> str:
    if not value:
        return "unknown_game"
    lowered = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_game"

def _normalize_keys(keys) -> List[str]:
    if not keys:
        return []
    if isinstance(keys, str):
        keys = [keys]
    normalized = []
    for key in keys:
        if key is None:
            continue
        value = str(key).strip().lower()
        if value:
            normalized.append(value)
    return normalized

def _dedupe_keys(keys: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered

@dataclass
class FrameRecord:
    timestamp: float
    frame: np.ndarray


def _handle_signal(_signum, _frame):
    stop_event.set()


def decode_frame(payload: dict) -> Optional[np.ndarray]:
    data = get_frame_bytes(payload)
    if not data:
        return None
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


class GameOnboardingAgent:
    def __init__(self):
        self.lock = threading.Lock()
        self.client = mqtt.Client(client_id="game_onboarding", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.frames: Deque[FrameRecord] = deque(maxlen=300)
        self.latest_state: Dict = {}
        self.latest_cursor: Dict = {}
        self.schema: Dict[str, object] = {}
        self.control_results: List[Dict[str, object]] = []
        self.running = True
        self.analysis_done = threading.Event()
        self.profile: Dict[str, object] = {}
        self.profile_status: str = "unknown"
        self.game_id: str = "unknown_game"
        self.llm_status: str = "not_requested"

    # MQTT wiring -------------------------------------------------------------
    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            topics = [(FRAME_TOPIC, 0), (SCENE_TOPIC, 0), (CURSOR_TOPIC, 0)]
            if GAME_IDENTITY_TOPIC:
                topics.append((GAME_IDENTITY_TOPIC, 0))
            client.subscribe(topics)
            logger.info("Onboarding connected, topics: %s", [t for t, _ in topics])
        else:
            logger.error("Onboarding connect failed code=%s", _as_int(rc))

    def _on_message(self, _client, _userdata, msg):
        with self.lock:
            if msg.topic == FRAME_TOPIC:
                frame = decode_frame(json.loads(msg.payload.decode("utf-8", "ignore")))
                if frame is not None:
                    self.frames.append(FrameRecord(time.time(), frame))
            elif msg.topic == SCENE_TOPIC:
                try:
                    data = json.loads(msg.payload.decode("utf-8", "ignore"))
                except Exception:
                    return
                if isinstance(data, dict):
                    self.latest_state = data
                    if data.get("game_id"):
                        self.game_id = _normalize_game_id(str(data.get("game_id")))
            elif msg.topic == CURSOR_TOPIC:
                try:
                    data = json.loads(msg.payload.decode("utf-8", "ignore"))
                except Exception:
                    return
                if isinstance(data, dict):
                    self.latest_cursor = data
            elif GAME_IDENTITY_TOPIC and msg.topic == GAME_IDENTITY_TOPIC:
                try:
                    data = json.loads(msg.payload.decode("utf-8", "ignore"))
                except Exception:
                    return
                if isinstance(data, dict):
                    raw_id = data.get("game_id") or data.get("app_name") or data.get("window_title") or ""
                    normalized = _normalize_game_id(str(raw_id))
                    if normalized and normalized != "unknown_game":
                        self.game_id = normalized

    # Analysis ----------------------------------------------------------------
    def _collect_phase_a(self):
        logger.info("Phase A start")
        start = time.time()
        while time.time() - start < PHASE_A_SECONDS:
            if stop_event.is_set():
                logger.info("Onboarding aborted")
                return
            time.sleep(0.2)
        logger.info("Phase A done, frames: %s", len(self.frames))

    def _compute_layout(self):
        with self.lock:
            if len(self.frames) < 2:
                return {}
            # Copy frames to a local list to avoid holding the lock during computation
            local_frames = list(self.frames)

        resized = []
        for record in local_frames:
            frame = cv2.resize(record.frame, (FRAME_DOWNSCALE, FRAME_DOWNSCALE), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized.append(gray)
        stack = np.stack(resized, axis=0)
        variance = stack.var(axis=0)
        dyn_mask = (variance > VARIANCE_THRESHOLD).astype(np.uint8)
        static_mask = 1 - dyn_mask
        play_bbox = self._mask_to_bbox(dyn_mask)
        hud_boxes = self._find_hud_boxes(static_mask)
        hud_entries = []
        for box in hud_boxes:
            hud_entries.append({
                "x": round(box[0], 4),
                "y": round(box[1], 4),
                "w": round(box[2], 4),
                "h": round(box[3], 4),
            })
        layout = {
            "play_area": play_bbox,
            "hud_candidates": hud_entries,
        }
        logger.info("Layout computed: %s", layout)
        return layout

    def _detect_game_id(self):
        """Infer game_id from state text or LLM."""
        if GAME_ID_OVERRIDE:
            self.game_id = _normalize_game_id(GAME_ID_OVERRIDE)
            return
        # Prefer explicit game_id if present
        with self.lock:
            explicit = self.latest_state.get("game_id")
            texts = list(self.latest_state.get("text") or [])[:20]
        if explicit:
            normalized = _normalize_game_id(str(explicit))
            if normalized not in {"unknown_game", "unknown"}:
                self.game_id = normalized
                return
        if REQUEST_LLM_GAME_ID:
            guessed, status = guess_game_id(self.game_id, texts)
            if guessed:
                self.game_id = _normalize_game_id(guessed)
                self.llm_status = status
        self.game_id = _normalize_game_id(self.game_id)

    def _mask_to_bbox(self, mask: np.ndarray) -> Dict[str, float]:
        coords = cv2.findNonZero(mask)
        if coords is None:
            return {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
        x, y, w, h = cv2.boundingRect(coords)
        size = mask.shape[0]
        return {
            "x": round(x / size, 4),
            "y": round(y / size, 4),
            "w": round(w / size, 4),
            "h": round(h / size, 4),
        }

    def _find_hud_boxes(self, mask: np.ndarray) -> List[Tuple[float, float, float, float]]:
        size = mask.shape[0]
        mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_norm = (w * h) / (size * size)
            if HUD_AREA_MIN <= area_norm <= HUD_AREA_MAX:
                boxes.append((x / size, y / size, w / size, h / size))
        return boxes

    # Control probing ---------------------------------------------------------
    def _probe_controls(self, probes=None):
        logger.info("Control probe start")
        probes = probes or CONTROL_PROBES
        results = []
        for name, action, label in probes:
            if stop_event.is_set():
                return results
            before = self._latest_frame_gray()
            baseline = self._capture_state_snapshot()
            self._publish_action(action)
            time.sleep(CONTROL_SETTLE_SEC)
            after = self._latest_frame_gray()
            diff_score = self._frame_difference(before, after)
            result = {
                "input": name,
                "label": label,
                "diff_score": diff_score,
                "confidence": min(1.0, diff_score / (CONTROL_DIFF_THRESHOLD * 2.0)),
            }
            result.update(baseline)
            self.control_results.append(result)
            results.append(result)
            logger.info("Control probe result: %s", result)
            time.sleep(POST_PROBE_SEC)
        return results

    def _frame_difference(self, before: Optional[np.ndarray], after: Optional[np.ndarray]) -> float:
        if before is None or after is None:
            return 0.0
        before = cv2.resize(before, (FRAME_DOWNSCALE, FRAME_DOWNSCALE))
        after = cv2.resize(after, (FRAME_DOWNSCALE, FRAME_DOWNSCALE))
        diff = cv2.absdiff(before, after)
        return float(diff.mean())

    def _latest_frame_gray(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.frames:
                return None
            frame = self.frames[-1].frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _capture_state_snapshot(self) -> Dict[str, object]:
        with self.lock:
            state = self.latest_state.get("text") or []
        return {
            "scene_text": list(state)[:20],
        }

    def _publish_action(self, payload: dict):
        message = dict(payload)
        message["ok"] = True
        self.client.publish(ACT_TOPIC, json.dumps(message))

    # Schema synthesis --------------------------------------------------------
    def _profile_to_controls(self, profile: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        """Convert a control profile into schema-style control entries."""
        controls: Dict[str, Dict[str, object]] = {}
        if profile.get("allow_mouse_move"):
            controls["mouse_move"] = {"input": "mouse_move", "confidence": 0.8, "diff_score": 0.0}
        if profile.get("allow_primary"):
            controls["basic_attack"] = {"input": "click_primary", "confidence": 0.8, "diff_score": 0.0}
        if profile.get("allow_secondary"):
            controls["alt_attack"] = {"input": "click_secondary", "confidence": 0.7, "diff_score": 0.0}
        for key in profile.get("allowed_keys", []):
            label = f"key_{key.lower()}"
            controls[label] = {"input": f"key_{key.lower()}", "confidence": 0.6, "diff_score": 0.0}
        return controls

    def _build_schema(self, layout: Dict[str, object]):
        controls = {}
        for result in self.control_results:
            controls[result["label"]] = {
                "input": result["input"],
                "confidence": round(min(1.0, result.get("confidence", 0.0)), 3),
                "diff_score": round(result.get("diff_score", 0.0), 2),
            }

        if not controls:
            controls = self._profile_to_controls(self.profile)

        with self.lock:
            texts = self.latest_state.get("text") or []
            frames_collected = len(self.frames)

        hud_signals = self._extract_signals(texts)
        schema = {
            "game_id": self.game_id,
            "ui_layout": layout,
            "controls": controls,
            "signals": hud_signals,
            "profile": self.profile,
            "profile_status": self.profile_status,
            "llm_status": self.llm_status,
            "notes": [
                "Automatic onboarding pass",
                f"frames_collected={frames_collected}",
                f"profile_status={self.profile_status}",
                f"llm_status={self.llm_status}",
            ],
        }
        self.schema = schema
        self._publish_schema(schema)

    def _extract_signals(self, texts: List[str]) -> Dict[str, List[str]]:
        danger_tokens = []
        progress_tokens = []
        for entry in texts:
            lowered = entry.lower()
            if any(token in lowered for token in ("dead", "died", "game over")):
                danger_tokens.append(entry)
            if any(token in lowered for token in ("level", "quest", "complete", "victory")):
                progress_tokens.append(entry)
        return {
            "danger_signals": danger_tokens[:5],
            "progress_signals": progress_tokens[:5],
        }

    def _publish_schema(self, schema: Dict[str, object]):
        payload = {"ok": True, "schema": schema, "timestamp": time.time()}
        self.client.publish(SCHEMA_TOPIC, json.dumps(payload))
        if MEM_STORE_TOPIC:
            store_message = {
                "op": "set",
                "key": "game_schema",
                "value": schema,
            }
            self.client.publish(MEM_STORE_TOPIC, json.dumps(store_message))
        summary = {
            "event": "onboarding_summary",
            "play_area": schema.get("ui_layout", {}).get("play_area"),
            "controls": list(schema.get("controls", {}).keys()),
        }
        logger.info("Schema summary: %s", summary)

    # Main -------------------------------------------------------------------
    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        try:
            self._collect_phase_a()
            if stop_event.is_set(): return
            layout = self._compute_layout()
            if stop_event.is_set(): return
            self._detect_game_id()
            # Resolve control profile
            llm_confident = False
            texts = []
            with self.lock:
                texts = list(self.latest_state.get("text") or [])[:20]
            local_profile = load_profile(self.game_id)
            if PREFER_LOCAL_PROFILE and local_profile:
                self.profile = local_profile
                self.profile_status = "loaded"
            else:
                if REQUEST_LLM_PROFILE:
                    llm_profile, status = fetch_control_profile(self.game_id, texts)
                    self.llm_status = status
                    if llm_profile:
                        llm_profile["game_id"] = self.game_id
                        llm_profile.setdefault("profile_version", 1)
                        if FORCE_MOUSE_MOVE and not llm_profile.get("allow_mouse_move", True):
                            llm_profile["allow_mouse_move"] = True
                            llm_profile.setdefault("notes", []).append("allow_mouse_move forced")
                        safe_keys = _normalize_keys(llm_profile.get("allowed_keys_safe") or llm_profile.get("allowed_keys") or [])
                        extended_keys = _normalize_keys(llm_profile.get("allowed_keys_extended") or [])
                        safe_keys = [k for k in safe_keys if k not in SAFE_KEY_BLACKLIST]
                        extended_keys = [k for k in extended_keys if k not in SAFE_KEY_BLACKLIST]
                        mouse_mode = str(llm_profile.get("mouse_mode") or "").lower()
                        if mouse_mode == "click_to_move":
                            if safe_keys and all(k in {"w", "a", "s", "d"} for k in safe_keys):
                                extended_keys = _dedupe_keys(extended_keys + safe_keys)
                                safe_keys = []
                            if not safe_keys and extended_keys:
                                promote = [k for k in extended_keys if k in ARPG_DEFAULT_KEYS]
                                if promote:
                                    safe_keys = promote
                                    extended_keys = [k for k in extended_keys if k not in promote]
                        llm_profile["allowed_keys"] = _dedupe_keys(safe_keys)
                        llm_profile["allowed_keys_extended"] = _dedupe_keys(extended_keys)
                        self.profile = llm_profile
                        self.profile_status = "llm_generated"
                        logger.info(
                            "LLM profile loaded status=%s safe_keys=%s extended_keys=%s allow_mouse_move=%s",
                            status,
                            len(llm_profile.get("allowed_keys") or []),
                            len(llm_profile.get("allowed_keys_extended") or []),
                            llm_profile.get("allow_mouse_move"),
                        )
                        llm_confident = len(llm_profile.get("allowed_keys") or []) >= LLM_CONF_MIN_KEYS
                    else:
                        logger.warning("LLM profile unavailable status=%s", status)
                        llm_confident = False
                if not llm_confident:
                    if local_profile:
                        self.profile = local_profile
                        self.profile_status = "loaded"
                    else:
                        self.profile = safe_profile(self.game_id)
                        self.profile_status = "safe_fallback"

            # Normalize profile keys
            safe_keys = _normalize_keys(self.profile.get("allowed_keys") or [])
            extended_keys = _normalize_keys(self.profile.get("allowed_keys_extended") or [])
            safe_keys = [k for k in safe_keys if k not in SAFE_KEY_BLACKLIST]
            extended_keys = [k for k in extended_keys if k not in SAFE_KEY_BLACKLIST]
            self.profile["allowed_keys"] = _dedupe_keys(safe_keys)
            if extended_keys:
                self.profile["allowed_keys_extended"] = _dedupe_keys(extended_keys)

            # Probe only if we lack a confident schema/profile; use the safe subset.
            should_probe = (
                not DISABLE_PROBES
                and not llm_confident
                and self.profile_status not in {"loaded", "llm_generated"}
            )
            if PROBE_WITHOUT_PROFILE and not llm_confident and not DISABLE_PROBES:
                should_probe = True
            if should_probe:
                self._probe_controls(SAFE_PROBES)
            # If we have a confident profile (loaded or LLM) and allowed_keys, run a limited validation
            validate_keys = (
                not DISABLE_PROBES
                and (llm_confident or self.profile_status in {"loaded", "llm_generated"})
            )
            allowed_keys = [k.lower() for k in self.profile.get("allowed_keys", []) if k]
            allowed_keys = [k for k in allowed_keys if k not in SAFE_KEY_BLACKLIST]
            if validate_keys and allowed_keys:
                key_probes = [("KEY_" + k.upper(), {"action": "key_press", "key": k}, f"key_{k}") for k in allowed_keys]
                probes = SAFE_PROBES + key_probes
                self._probe_controls(probes)
            # Validate extended keys via probes before activating them
            if EXTENDED_PROBE and validate_keys:
                extended_keys = [k for k in self.profile.get("allowed_keys_extended", []) if k]
                extended_keys = [k for k in extended_keys if k not in SAFE_KEY_BLACKLIST]
                extended_keys = [k for k in extended_keys if k not in allowed_keys]
                if extended_keys:
                    ext_probes = [("KEY_" + k.upper(), {"action": "key_press", "key": k}, f"key_{k}") for k in extended_keys]
                    results = self._probe_controls(ext_probes)
                    verified = []
                    for result in results or []:
                        if float(result.get("diff_score") or 0.0) < EXTENDED_DIFF_MIN:
                            continue
                        label = str(result.get("label") or "")
                        if label.startswith("key_"):
                            verified.append(label.replace("key_", "", 1))
                    verified = _dedupe_keys([k for k in verified if k])
                    if verified:
                        self.profile["allowed_keys"] = _dedupe_keys(self.profile.get("allowed_keys", []) + verified)
                    self.profile["allowed_keys_extended_verified"] = verified
            upsert_profile(self.profile)
            if stop_event.is_set(): return
            self._build_schema(layout)
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            self.running = False


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = GameOnboardingAgent()
    agent.run()


if __name__ == "__main__":
    main()
