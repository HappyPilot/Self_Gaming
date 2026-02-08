#!/usr/bin/env python3
"""Policy agent that blends heuristic/goap behavior with teacher and PPO weights."""
from __future__ import annotations

import difflib
import math
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt
import torch
import torch.nn as nn

from models.backbone import Backbone
from utils.embedding_projector import EmbeddingProjector
from utils.latency import emit_control_metric, emit_latency, get_float_env, get_sla_ms
try:
    from .control_profile import load_profile, safe_profile
except ImportError:
    from control_profile import load_profile, safe_profile


def _normalize_phrase(text: str) -> str:
    if not text:
        return ""
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(normalized.split())


def _normalize_game_id(value: str) -> str:
    if not value:
        return "unknown_game"
    lowered = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_game"


def _candidate_chunks(cleaned: str):
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
    if not text:
        return ""
    filtered = [ch for ch in text.lower() if ch.isalnum()]
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

logging.basicConfig(level=os.getenv("POLICY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("policy_agent")


def _parse_env_list(value: str) -> Set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _parse_action_list(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OBS_TOPIC = os.getenv("OBS_TOPIC", "scene/state")
SIM_TOPIC = os.getenv("SIM_TOPIC", "sim_core/state")
GOAP_TOPIC = os.getenv("GOAP_TASK_TOPIC", "goap/tasks")
CONTROL_TOPIC = os.getenv("ACT_TOPIC", "control/keys")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
TEACHER_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action")
CHECKPOINT_TOPIC = os.getenv("CHECKPOINT_TOPIC", "train/checkpoints")
PROGRESS_TOPIC = os.getenv("PROGRESS_TOPIC", "progress/status")
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
GAME_SCHEMA_TOPIC = os.getenv("GAME_SCHEMA_TOPIC", "game/schema")
CONTROL_PROFILE_GAME_ID = os.getenv("CONTROL_PROFILE_GAME_ID") or os.getenv("RECORDER_GAME_ID") or os.getenv("GAME_ID") or "unknown_game"
THRESH = float(os.getenv("THRESH", "120.0"))
DEBOUNCE = float(os.getenv("DEBOUNCE", "0.25"))
SLA_STAGE_POLICY_MS = get_sla_ms("SLA_STAGE_POLICY_MS")
SLA_TICK_MS = get_sla_ms("SLA_TICK_MS")
SLA_JERK_MAX = get_float_env("SLA_JERK_MAX")
CONTROL_READY_WINDOW = max(1, int(os.getenv("CONTROL_READY_WINDOW", "50")))
CONTROL_METRIC_SAMPLE_EVERY = max(1, int(os.getenv("CONTROL_METRIC_SAMPLE_EVERY", "5")))
TEACHER_ALPHA_START = float(os.getenv("TEACHER_ALPHA_START", "1.0"))
TEACHER_DECAY_STEPS = int(os.getenv("TEACHER_ALPHA_DECAY_STEPS", "500"))
MIN_ALPHA = float(os.getenv("TEACHER_ALPHA_MIN", "0.0"))
TEACHER_TARGET_PRIORITY = os.getenv("TEACHER_TARGET_PRIORITY", "1") != "0"
TEACHER_TARGET_AUTOCLICK = os.getenv("TEACHER_TARGET_AUTOCLICK", "0") != "0"
TEACHER_TARGET_AUTOCLICK_COOLDOWN = float(os.getenv("TEACHER_TARGET_AUTOCLICK_COOLDOWN", "2.0"))
TEACHER_TARGET_AUTOCLICK_SAME_TOL = float(os.getenv("TEACHER_TARGET_AUTOCLICK_SAME_TOL", "0.04"))
MOUSE_RANGE = int(os.getenv("POLICY_MOUSE_RANGE", "60"))
MIN_MOUSE_DELTA = int(os.getenv("POLICY_MIN_MOUSE_DELTA", "8"))
CLICK_COOLDOWN = float(os.getenv("POLICY_CLICK_COOLDOWN", "0.75"))
NUMERIC_DIM = 32
OBJECT_HIST_DIM = 32
TEXT_EMBED_DIM = 64
BASE_NON_VISUAL_DIM = NUMERIC_DIM + OBJECT_HIST_DIM + TEXT_EMBED_DIM
EMBED_FEATURE_ENABLED = os.getenv("EMBED_FEATURE_ENABLED", "0") != "0"
EMBED_FEATURE_DIM = int(os.getenv("EMBED_FEATURE_DIM", "128"))
EMBED_FEATURE_SOURCE_DIM = int(os.getenv("EMBED_FEATURE_SOURCE_DIM", "768"))
EMBED_FEATURE_SEED = int(os.getenv("EMBED_FEATURE_SEED", "42"))
EMBED_FEATURE_LAYER_NORM = os.getenv("EMBED_FEATURE_LAYER_NORM", "1") != "0"
NON_VISUAL_DIM = BASE_NON_VISUAL_DIM + (EMBED_FEATURE_DIM if EMBED_FEATURE_ENABLED else 0)
FRAME_SHAPE = (3, int(os.getenv("POLICY_FRAME_HEIGHT", "96")), int(os.getenv("POLICY_FRAME_WIDTH", "54")))
BACKBONE_PATH = Path(os.getenv("POLICY_BACKBONE_PATH", "/mnt/ssd/models/backbone/backbone.pt"))
POLICY_HEAD_PATH = Path(os.getenv("POLICY_HEAD_PATH", "/mnt/ssd/models/heads/ppo/policy_head.pt"))
VALUE_HEAD_PATH = Path(os.getenv("POLICY_VALUE_HEAD_PATH", "/mnt/ssd/models/heads/ppo/value_head.pt"))
LABEL_MAP_PATH = Path(os.getenv("POLICY_LABEL_MAP_PATH", "/mnt/ssd/models/heads/ppo/label_map.json"))
POLICY_LABEL_MAP_OPTIONAL = os.getenv("POLICY_LABEL_MAP_OPTIONAL", "1") != "0"
POLICY_LABEL_MAP_FALLBACK_PATH = Path(os.getenv("POLICY_LABEL_MAP_FALLBACK_PATH", "config/label_map.default.json"))
POLICY_DEFAULT_ACTIONS = _parse_action_list(os.getenv("POLICY_DEFAULT_ACTIONS", ""))
HOT_RELOAD_ENABLED = os.getenv("POLICY_HOT_RELOAD", "1") != "0"
POLICY_LAZY_LOAD = os.getenv("POLICY_LAZY_LOAD", "1") != "0"
POLICY_LAZY_RETRY_SEC = float(os.getenv("POLICY_LAZY_RETRY_SEC", "120"))
POLICY_AUTO_RELOAD = os.getenv("POLICY_AUTO_RELOAD", "1") != "0"
POLICY_AUTO_RELOAD_SEC = float(os.getenv("POLICY_AUTO_RELOAD_SEC", "5"))
POLICY_AUTO_RELOAD_DELAY_SEC = float(os.getenv("POLICY_AUTO_RELOAD_DELAY_SEC", "1.0"))
LEARNING_STAGE = int(os.getenv("LEARNING_STAGE", "1"))
STAGE0_ACTION_INTERVAL = float(os.getenv("STAGE0_ACTION_INTERVAL", "1.5"))
STAGE0_SETTLE_SEC = float(os.getenv("STAGE0_SETTLE_SEC", "1.0"))
SCREEN_WIDTH = int(os.getenv("POLICY_SCREEN_WIDTH", "1920"))
SCREEN_HEIGHT = int(os.getenv("POLICY_SCREEN_HEIGHT", "1080"))
CURSOR_OFFSET_X = int(os.getenv("POLICY_OFFSET_X", "0"))
CURSOR_OFFSET_Y = int(os.getenv("POLICY_OFFSET_Y", "0"))
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
POLICY_USE_CURSOR = os.getenv("POLICY_USE_CURSOR", "1") != "0"
CURSOR_TIMEOUT_SEC = float(os.getenv("CURSOR_TIMEOUT_SEC", "1.2"))
CURSOR_TOLERANCE = float(os.getenv("CURSOR_TOLERANCE", "0.02"))
MOTION_ANCHOR_TOPIC = os.getenv("MOTION_ANCHOR_TOPIC", "vision/anchors")
MOTION_ANCHOR_TTL = float(os.getenv("MOTION_ANCHOR_TTL", "2.0"))
HOVER_VERIFY_ENABLED = os.getenv("POLICY_HOVER_VERIFY", "1") != "0"
HOVER_FUZZY_THRESHOLD = float(os.getenv("POLICY_HOVER_FUZZY_THRESHOLD", "0.65"))
HOVER_FAIL_LIMIT = int(os.getenv("POLICY_HOVER_FAIL_LIMIT", "5"))
HOVER_FAIL_WINDOW = float(os.getenv("POLICY_HOVER_FAIL_WINDOW", "2.0"))
DESKTOP_KEYWORDS = _parse_env_list(os.getenv("POLICY_DESKTOP_KEYWORDS", "desktop,finder,windows,taskbar"))
DESKTOP_PAUSE_SEC = float(os.getenv("POLICY_DESKTOP_PAUSE_SEC", "3.0"))
SHOP_HOVER_BLOCK_ENABLED = os.getenv("POLICY_SHOP_HOVER_BLOCK", "1") != "0"
POLICY_FORBIDDEN_TAGS = _parse_env_list(os.getenv("POLICY_FORBIDDEN_TAGS", "shop_button"))
POLICY_FORBIDDEN_TEXTS = _parse_env_list(os.getenv("POLICY_FORBIDDEN_TEXTS", "shop"))
SHOP_SUPPRESS = os.getenv("SHOP_SUPPRESS", "0") != "0"
SHOP_SUPPRESS_DEATH_ONLY = os.getenv("SHOP_SUPPRESS_DEATH_ONLY", "1") != "0"
FEEDBACK_SETTLE_SEC = float(os.getenv("POLICY_FEEDBACK_SETTLE_SEC", "1.0"))
POLICY_GAME_KEYWORDS = _parse_env_list(
    os.getenv("POLICY_GAME_KEYWORDS", "health,hp,mana,xp,score,level,ammo,inventory,quest,map,target,enemy,boss,start,menu")
)
POLICY_REQUIRE_IN_GAME = os.getenv("POLICY_REQUIRE_IN_GAME", "0") != "0"
POLICY_REQUIRE_IN_GAME_STRICT = os.getenv("POLICY_REQUIRE_IN_GAME_STRICT", "0") != "0"
POLICY_REQUIRE_EMBEDDINGS = os.getenv("POLICY_REQUIRE_EMBEDDINGS", "0") != "0"
POLICY_EMBED_MAX_AGE_SEC = float(os.getenv("POLICY_EMBED_MAX_AGE_SEC", "5.0"))
POLICY_MEANINGFUL_FALLBACK = os.getenv("POLICY_MEANINGFUL_FALLBACK", "0") != "0"
POLICY_PREFER_TARGETS = os.getenv("POLICY_PREFER_TARGETS", "1") != "0"
POLICY_RANDOM_FALLBACK = os.getenv("POLICY_RANDOM_FALLBACK", "1") != "0"
PROACTIVE_IDLE_CHANCE = float(os.getenv("PROACTIVE_IDLE_CHANCE", "0.4"))
PROACTIVE_IDLE_RADIUS = float(os.getenv("PROACTIVE_IDLE_RADIUS", "0.08"))
PROACTIVE_IDLE_MOVE_PROB = float(os.getenv("PROACTIVE_IDLE_MOVE_PROB", "0.7"))
POLICY_USE_OCR_TARGETS = os.getenv("POLICY_USE_OCR_TARGETS", "1") != "0"
POLICY_ENEMY_SKILL_SUBSTITUTE = os.getenv("POLICY_ENEMY_SKILL_SUBSTITUTE", "1") != "0"
POLICY_ENEMY_SKILL_MIN_INTERVAL = float(os.getenv("POLICY_ENEMY_SKILL_MIN_INTERVAL", "0.4"))
POLICY_ENEMY_MIN_CONF = float(os.getenv("POLICY_ENEMY_MIN_CONF", "0.12"))
POLICY_COMBAT_AIM = os.getenv("POLICY_COMBAT_AIM", "1") != "0"
POLICY_SKILL_EPS = float(os.getenv("POLICY_SKILL_EPS", "0.15"))
POLICY_SKILL_FEEDBACK_SEC = float(os.getenv("POLICY_SKILL_FEEDBACK_SEC", "0.7"))
POLICY_ENEMY_CLUSTER_MIN = int(os.getenv("POLICY_ENEMY_CLUSTER_MIN", "3"))
POLICY_ENEMY_BAR_MIN_DELTA = float(os.getenv("POLICY_ENEMY_BAR_MIN_DELTA", "0.002"))
TEACHER_KEY_CONFIRM_HITS = int(os.getenv("TEACHER_KEY_CONFIRM_HITS", "1"))
POLICY_INCLUDE_EXTENDED_KEYS = os.getenv("POLICY_INCLUDE_EXTENDED_KEYS", "1") != "0"
POLICY_EXTENDED_KEYS_ALLOW = {
    item.strip().lower()
    for item in os.getenv("POLICY_EXTENDED_KEYS_ALLOW", "q,w,e,r,1,2,3,4,5").split(",")
    if item.strip()
}
POLICY_FALLBACK_SKILL_KEYS = [
    item.strip().lower()
    for item in os.getenv("POLICY_FALLBACK_SKILL_KEYS", "q,w,e,r,1,2,3,4,5").split(",")
    if item.strip()
]
POLICY_USE_UI_LAYOUT = os.getenv("POLICY_USE_UI_LAYOUT", "1") != "0"
POLICY_SKILL_KEYS = [item.strip().lower() for item in os.getenv("POLICY_SKILL_KEYS", "q,w,e,r,1,2,3,4,5").split(",") if item.strip()]
POLICY_DIALOG_SCORE_MIN = float(os.getenv("POLICY_DIALOG_SCORE_MIN", "0.01"))
POLICY_INTENT_ENABLED = os.getenv("POLICY_INTENT_ENABLED", "1") != "0"
POLICY_INTENT_MIN_SEC = float(os.getenv("POLICY_INTENT_MIN_SEC", "1.0"))
POLICY_INTENT_DIALOG_SCORE = float(os.getenv("POLICY_INTENT_DIALOG_SCORE", "0.01"))
POLICY_INTENT_ENEMY_SCORE = float(os.getenv("POLICY_INTENT_ENEMY_SCORE", "0.01"))
POLICY_INTENT_INTERACT_SCORE = float(os.getenv("POLICY_INTENT_INTERACT_SCORE", "0.01"))
POLICY_INTENT_NAV_SCORE = float(os.getenv("POLICY_INTENT_NAV_SCORE", "0.01"))
POLICY_INTENT_COMBAT_LABELS = _parse_env_list(os.getenv("POLICY_INTENT_COMBAT_LABELS", "enemy,boss,monster,hostile"))
POLICY_INTENT_INTERACT_LABELS = _parse_env_list(
    os.getenv("POLICY_INTENT_INTERACT_LABELS", "loot,chest,npc,vendor,shop,door,gate,exit,dialog_button,ui_button,item,pickup,corpse")
)
POLICY_INTENT_NAV_LABELS = _parse_env_list(
    os.getenv("POLICY_INTENT_NAV_LABELS", "waypoint,portal,quest,quest_marker,objective,objective_marker,door,gate,exit")
)
RESPAWN_TEXTS = _parse_env_list(
    os.getenv("RESPAWN_TRIGGER_TEXTS", "resurrect at checkpoint,resurrect in town,resurrect")
)
RESPAWN_TARGET_X = float(os.getenv("RESPAWN_TARGET_X", os.getenv("GOAP_DIALOG_BUTTON_X", "0.5")))
RESPAWN_TARGET_Y = float(os.getenv("RESPAWN_TARGET_Y", os.getenv("GOAP_DIALOG_BUTTON_Y", "0.82")))
RESPAWN_COOLDOWN = float(os.getenv("RESPAWN_COOLDOWN_SEC", "2.0"))
RESPAWN_FUZZY_THRESHOLD = float(os.getenv("RESPAWN_FUZZY_THRESHOLD", "0.6"))
RESPAWN_SKELETON_THRESHOLD = float(os.getenv("RESPAWN_SKELETON_THRESHOLD", "0.72"))
RESPAWN_SKELETON_MIN_LEN = int(os.getenv("RESPAWN_SKELETON_MIN_LEN", "6"))
RESPAWN_DEBUG = os.getenv("RESPAWN_DEBUG", "0") != "0"
RESPAWN_MACRO_MOVE_STEPS = int(os.getenv("RESPAWN_MACRO_MOVE_STEPS", "2"))
RESPAWN_MACRO_MOVE_DELAY = float(os.getenv("RESPAWN_MACRO_MOVE_DELAY", "0.1"))
RESPAWN_MACRO_CLICK_COUNT = int(os.getenv("RESPAWN_MACRO_CLICK_COUNT", "3"))
RESPAWN_MACRO_CLICK_DELAY = float(os.getenv("RESPAWN_MACRO_CLICK_DELAY", "0.15"))
RESPAWN_MACRO_PRE_CLICK_DELAY = float(os.getenv("RESPAWN_MACRO_PRE_CLICK_DELAY", "1.5"))
RESPAWN_MACRO_SETTLE_SEC = float(os.getenv("RESPAWN_MACRO_SETTLE_SEC", "1.5"))
POLICY_SCENE_MAX_AGE = float(os.getenv("POLICY_SCENE_MAX_AGE", "3.0"))
ACTIVE_TARGET_TTL = float(os.getenv("POLICY_TARGET_TTL", "1.5"))
MIN_TARGET_DELTA = float(os.getenv("POLICY_TARGET_MIN_DELTA", "2.0"))
POLICY_SCENE_BRIGHT_THRESHOLD = float(os.getenv("POLICY_SCENE_BRIGHT_THRESHOLD", "0.65"))
POLICY_FORBIDDEN_COOLDOWN = float(os.getenv("POLICY_FORBIDDEN_COOLDOWN", "3.0"))
HOVER_DEBUG = os.getenv("POLICY_HOVER_DEBUG", "0") != "0"
TEACHER_TARGET_COORD_RE = re.compile(
    r"(?:target|coord(?:inate)?s?)\s*[:=]?\s*[\(\[]?\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*[\)\]]?",
    re.IGNORECASE,
)
TEACHER_PAREN_COORD_RE = re.compile(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
TEACHER_TARGET_LABEL_RE = re.compile(r"target_hint\s*:?\s*([^|\n\r]+)", re.IGNORECASE)

POLICY_EXPLORATION_ENABLED = os.getenv("POLICY_EXPLORATION_ENABLED", "1") != "0"
POLICY_EXPLORATION_REWARD_TIMEOUT_SEC = float(os.getenv("POLICY_EXPLORATION_REWARD_TIMEOUT_SEC", "180"))
POLICY_EXPLORATION_MIN_REWARD = float(os.getenv("POLICY_EXPLORATION_MIN_REWARD", "0.05"))
POLICY_EXPLORATION_LOCATION_AGE_SEC = float(os.getenv("POLICY_EXPLORATION_LOCATION_AGE_SEC", "90"))
POLICY_EXPLORATION_OBJECT_NEW_RATE_MAX = float(os.getenv("POLICY_EXPLORATION_OBJECT_NEW_RATE_MAX", "0.02"))
POLICY_EXPLORATION_REPEAT_WINDOW = max(1, int(os.getenv("POLICY_EXPLORATION_REPEAT_WINDOW", "8")))
POLICY_EXPLORATION_REPEAT_MIN = max(1, int(os.getenv("POLICY_EXPLORATION_REPEAT_MIN", "6")))
POLICY_EXPLORATION_COOLDOWN_SEC = float(os.getenv("POLICY_EXPLORATION_COOLDOWN_SEC", "120"))
POLICY_EXPLORATION_BURST_ACTIONS = int(os.getenv("POLICY_EXPLORATION_BURST_ACTIONS", "6"))
POLICY_EXPLORATION_ALLOW_CLICK = os.getenv("POLICY_EXPLORATION_ALLOW_CLICK", "0") != "0"
POLICY_EXPLORATION_KEYS = _parse_env_list(os.getenv("POLICY_EXPLORATION_KEYS", ""))
POLICY_EXPLORATION_KEYS_FROM_PROFILE = os.getenv("POLICY_EXPLORATION_KEYS_FROM_PROFILE", "0") != "0"
POLICY_EXPLORATION_MARGIN = float(os.getenv("POLICY_EXPLORATION_MARGIN", "0.08"))
POLICY_EXPLORATION_MOUSE_RANGE = int(os.getenv("POLICY_EXPLORATION_MOUSE_RANGE", str(MOUSE_RANGE * 2)))
POLICY_EXPLORATION_KEY_BURST = os.getenv("POLICY_EXPLORATION_KEY_BURST", "0") != "0"
POLICY_EXPLORATION_KEY_BURST_COUNT = int(os.getenv("POLICY_EXPLORATION_KEY_BURST_COUNT", "2"))
POLICY_EXPLORATION_KEY_BURST_COOLDOWN_SEC = float(os.getenv("POLICY_EXPLORATION_KEY_BURST_COOLDOWN_SEC", "600"))
POLICY_CLICK_TO_MOVE = os.getenv("POLICY_CLICK_TO_MOVE", "0") != "0"

DEFAULT_ACTIONS = [
    "wait",
    "mouse_move",
    "click_primary",
    "click_secondary",
    "mouse_hold",
    "mouse_release",
    "key_w",
    "key_a",
    "key_s",
    "key_d",
    "key_q",
    "key_e",
    "key_r",
    "key_t",
    "key_f",
    "key_1",
    "key_2",
    "key_3",
    "key_4",
    "key_5",
    "key_space",
    "key_tab",
    "key_enter",
    "key_escape",
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_PROJECTOR = (
    EmbeddingProjector(
        in_dim=EMBED_FEATURE_SOURCE_DIM,
        out_dim=EMBED_FEATURE_DIM,
        seed=EMBED_FEATURE_SEED,
        use_layer_norm=EMBED_FEATURE_LAYER_NORM,
    )
    if EMBED_FEATURE_ENABLED
    else None
)


def compute_cursor_motion(
    x_norm: float,
    y_norm: float,
    cursor_x_norm: float,
    cursor_y_norm: float,
    width: int,
    height: int,
    offset_x: int = 0,
    offset_y: int = 0,
):
    """Project normalized coordinates into pixels and return deltas."""

    target_px = (
        int(round(x_norm * width + offset_x)),
        int(round(y_norm * height + offset_y)),
    )
    cursor_px = (
        int(round(cursor_x_norm * width + offset_x)),
        int(round(cursor_y_norm * height + offset_y)),
    )
    delta = (target_px[0] - cursor_px[0], target_px[1] - cursor_px[1])
    return target_px, cursor_px, delta


def _coerce_label_map(raw: object) -> Dict[str, int]:
    if isinstance(raw, dict):
        mapping = {}
        for action, idx in raw.items():
            try:
                mapping[str(action)] = int(idx)
            except (TypeError, ValueError):
                continue
        return mapping
    if isinstance(raw, list):
        mapping = {}
        for idx, action in enumerate(raw):
            if action:
                mapping[str(action)] = idx
        return mapping
    return {}


def _load_label_map_from_path(path: Optional[Path]) -> Tuple[Dict[str, int], str]:
    if not path:
        return {}, "missing"
    if not path.exists():
        return {}, f"missing:{path}"
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load label_map from %s: %s", path, exc)
        return {}, f"invalid:{path}"
    label_map = _coerce_label_map(raw)
    if not label_map:
        return {}, f"empty:{path}"
    return label_map, f"path:{path}"


def _fallback_actions(num_classes: Optional[int]) -> Tuple[List[str], str]:
    if POLICY_DEFAULT_ACTIONS:
        actions = list(POLICY_DEFAULT_ACTIONS)
        source = "env"
    else:
        actions = []
        source = "builtin"
        if POLICY_LABEL_MAP_FALLBACK_PATH.exists():
            label_map, origin = _load_label_map_from_path(POLICY_LABEL_MAP_FALLBACK_PATH)
            if label_map:
                actions = [action for action, _idx in sorted(label_map.items(), key=lambda item: item[1])]
                source = origin
        if not actions:
            actions = list(DEFAULT_ACTIONS)
            source = "builtin"
    if num_classes is not None and num_classes > 0:
        if len(actions) < num_classes:
            actions.extend([f"action_{idx}" for idx in range(len(actions), num_classes)])
        elif len(actions) > num_classes:
            actions = actions[:num_classes]
    return actions, source


def _build_idx_to_action(label_map: Dict[str, int], num_classes: int) -> Dict[int, str]:
    idx_to_action: Dict[int, str] = {}
    for action, idx in label_map.items():
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_int < num_classes:
            idx_to_action[idx_int] = action
    for idx in range(num_classes):
        if idx not in idx_to_action:
            idx_to_action[idx] = f"action_{idx}"
    return idx_to_action

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
    text_entries = _gather_scene_texts(state)
    
    # 0: Mean brightness
    numeric[0] = float(state.get("mean", state.get("mean_brightness", 0.0)))
    # 1: Total object count
    numeric[1] = float(len(objects))
    # 2: Enemy count (from stats or manual count)
    stats = state.get("stats") or {}
    enemy_count = stats.get("enemy_count")
    if enemy_count is None:
        enemy_count = sum(1 for obj in objects if "enemy" in str(obj.get("class", "")).lower())
    numeric[2] = float(enemy_count)
    # 3: Loot count (from stats or manual count)
    loot_count = stats.get("loot_count")
    if loot_count is None:
        loot_count = sum(1 for obj in objects if "loot" in str(obj.get("class", "")).lower())
    numeric[3] = float(loot_count)
    # 4: Text entries count
    numeric[4] = float(len(text_entries))
    
    def normalize_stat(info):
        if not isinstance(info, dict):
            return 0.0
        cur = float(info.get("current", 0))
        # If it's a ratio (X/Y or X%), use simple division
        if not info.get("is_plain_value") and info.get("max", 0) > 0:
            return cur / float(info["max"])
        # Option 1: Percent+Log for plain values
        if cur <= 100:
            return cur / 100.0
        else:
            return min(1.0, math.log1p(cur) / math.log1p(1000))

    # 5, 6, 7: Generic Stats from resources
    resources = state.get("resources") or {}
    stat_values = []
    # Prioritize known important stats
    for key in ["life", "hp", "mana", "xp"]:
        if key in resources:
            stat_values.append(normalize_stat(resources[key]))
    
    # Fill remaining slots
    for key, info in resources.items():
        if key not in ["life", "hp", "mana", "xp"] and len(stat_values) < 3:
            stat_values.append(normalize_stat(info))
    
    for i, val in enumerate(stat_values[:3]):
        numeric[5 + i] = val
    
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
    text_offset = start + OBJECT_HIST_DIM
    vector[text_offset : text_offset + TEXT_EMBED_DIM] = text_bins
    if EMBED_FEATURE_ENABLED and EMBED_PROJECTOR is not None:
        embedding = state.get("embeddings") or state.get("embedding")
        projected = EMBED_PROJECTOR.project(embedding)
        if projected is not None:
            embed_tensor = torch.from_numpy(projected)
            embed_offset = BASE_NON_VISUAL_DIM
            vector[embed_offset : embed_offset + EMBED_FEATURE_DIM] = embed_tensor
    return vector


def _gather_scene_texts(state: Dict) -> List[str]:
    entries: List[str] = []
    base = state.get("text") or []
    if isinstance(base, list):
        entries.extend(str(entry) for entry in base if entry)
    text_zones = state.get("text_zones") or {}
    if isinstance(text_zones, dict):
        for zone in text_zones.values():
            if isinstance(zone, dict) and zone.get("text"):
                entries.append(str(zone.get("text")))
    resources = state.get("resources") or {}
    if isinstance(resources, dict):
        for name, info in resources.items():
            if not isinstance(info, dict):
                continue
            cur = info.get("current")
            maxv = info.get("max")
            if cur is None or maxv is None:
                continue
            entries.append(f"{name}:{cur}/{maxv}")
    return entries


class PolicyAgent:
    """Combines a baseline policy with teacher suggestions via annealing."""

    def __init__(self):
        self.state_lock = threading.Lock()
        self.client = mqtt.Client(client_id="policy", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.last_action_ts = 0.0
        self.last_enemy_skill_ts = 0.0
        self.last_combat_key: Optional[str] = None
        self.last_combat_score = None
        self.last_combat_ts = 0.0
        self.skill_stats: Dict[str, Dict[str, float]] = {}
        self.teacher_action: Optional[dict] = None
        self.latest_state: Optional[dict] = None
        self.current_task: Optional[dict] = None
        self.task_queue: Deque[dict] = deque()
        self.teacher_alpha_start = max(0.0, TEACHER_ALPHA_START)
        self.teacher_decay_steps = max(1, TEACHER_DECAY_STEPS)
        self.teacher_min_alpha = max(0.0, MIN_ALPHA)
        self.steps = 0
        self.last_label: Optional[str] = None
        self.profile_allowed_keys: Set[str] = set()
        self.profile_allowed_key_order: List[str] = []
        self.game_id = _normalize_game_id(CONTROL_PROFILE_GAME_ID)
        self.last_click_ts = 0.0
        self.rng = random.Random()
        self.model_lock = threading.Lock()
        self.reload_lock = threading.Lock()
        self.model: Optional[Dict] = None
        self.hot_reload_enabled = HOT_RELOAD_ENABLED
        self.lazy_load_enabled = POLICY_LAZY_LOAD
        self.lazy_retry_sec = max(0.0, POLICY_LAZY_RETRY_SEC)
        self.auto_reload_enabled = POLICY_AUTO_RELOAD
        self.auto_reload_sec = max(1.0, POLICY_AUTO_RELOAD_SEC)
        self.auto_reload_delay_sec = max(0.0, POLICY_AUTO_RELOAD_DELAY_SEC)
        self.last_checkpoint_signature: Optional[Tuple[Tuple[float, int], ...]] = None
        self.lazy_next_retry_at = 0.0
        self.lazy_last_retry_log_ts = 0.0
        self.model_load_attempted = False
        self.stage0_enabled = LEARNING_STAGE == 0
        self.stage0_pending = False
        self.stage0_reference = ""
        self.stage0_last_signature = ""
        self.stage0_wait_until = 0.0
        self.stage0_move_targets = {}
        self.cursor_x_norm = 0.5
        self.cursor_y_norm = 0.5
        self.cursor_detected_ts = 0.0
        self.player_center = [0.5, 0.5]
        self.scene_targets = []
        self.active_target = None
        self.active_target_expires = 0.0
        self.forbidden_tags = set(POLICY_FORBIDDEN_TAGS)
        self.forbidden_texts = set(POLICY_FORBIDDEN_TEXTS)
        self.shop_suppress_enabled = SHOP_SUPPRESS
        self.shop_suppress_death_only = SHOP_SUPPRESS_DEATH_ONLY
        self.feedback_pending = False
        self.last_json_error_ts = 0.0
        self.feedback_signature = ""
        self.game_keywords = set(POLICY_GAME_KEYWORDS)
        self.desktop_keywords = set(DESKTOP_KEYWORDS)
        self.last_scene_block_reason: Optional[str] = None
        self.forbidden_until = 0.0
        self.intent_enabled = POLICY_INTENT_ENABLED
        self.intent_min_sec = max(0.0, POLICY_INTENT_MIN_SEC)
        self.intent_dialog_score = POLICY_INTENT_DIALOG_SCORE
        self.intent_enemy_score = POLICY_INTENT_ENEMY_SCORE
        self.intent_interact_score = POLICY_INTENT_INTERACT_SCORE
        self.intent_nav_score = POLICY_INTENT_NAV_SCORE
        self.intent_combat_labels = {_normalize_phrase(label) for label in POLICY_INTENT_COMBAT_LABELS if label}
        self.intent_interact_labels = {_normalize_phrase(label) for label in POLICY_INTENT_INTERACT_LABELS if label}
        self.intent_nav_labels = {_normalize_phrase(label) for label in POLICY_INTENT_NAV_LABELS if label}
        self.current_intent: Optional[str] = None
        self.intent_reason: Optional[str] = None
        self.intent_last_change_ts = 0.0
        self.respawn_keywords = {_normalize_phrase(text) for text in RESPAWN_TEXTS if text}
        self.respawn_pending = False
        self.last_respawn_ts = 0.0
        self.forbidden_until = 0.0
        self.hover_failures: Dict[str, Dict[str, float]] = {}
        self.respawn_macro_queue: Deque[Tuple[float, Dict[str, object]]] = deque()
        self.respawn_macro_active = False
        self.respawn_macro_block_until = 0.0
        self.latest_cursor = {}
        self.control_ready_window: Deque[int] = deque(maxlen=CONTROL_READY_WINDOW)
        self.control_tick_count = 0
        self.last_control_action: Optional[dict] = None
        self.progress_state: Dict[str, object] = {}
        self.last_reward_ts = 0.0
        self.exploration_queue: Deque[Dict[str, object]] = deque()
        self.exploration_cooldown_until = 0.0
        self.play_area = None
        self.hud_boxes: List[Dict[str, float]] = []
        self.exploration_key_cooldown_until = 0.0
        self.recent_action_labels: Deque[str] = deque(maxlen=POLICY_EXPLORATION_REPEAT_WINDOW)
        self.last_autoclick = None
        self.consecutive_moves = 0
        self.motion_anchor: Optional[dict] = None
        if self.hot_reload_enabled and not self.lazy_load_enabled:
            self._initial_model_load()
            self.model_load_attempted = True
        if self.hot_reload_enabled and self.auto_reload_enabled:
            threading.Thread(target=self._auto_reload_loop, daemon=True).start()
            logger.info(
                "Policy auto-reload enabled | interval=%.1fs delay=%.1fs",
                self.auto_reload_sec,
                self.auto_reload_delay_sec,
            )
        logger.info(
            "Policy agent init | stage=%s stage0=%s screen=%sx%s",
            LEARNING_STAGE,
            self.stage0_enabled,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )
        logger.info(
            "Policy checkpoints | backbone=%s policy_head=%s label_map=%s exists=%s",
            BACKBONE_PATH,
            POLICY_HEAD_PATH,
            LABEL_MAP_PATH,
            LABEL_MAP_PATH.exists(),
        )

    # ------------------------------------------------------------------ Model
    def _initial_model_load(self):
        if BACKBONE_PATH.exists() and POLICY_HEAD_PATH.exists():
            logger.info("Attempting initial model load from %s", POLICY_HEAD_PATH.parent)
            self._reload_worker(
                {
                    "backbone_path": str(BACKBONE_PATH),
                    "policy_head_path": str(POLICY_HEAD_PATH),
                    "value_head_path": str(VALUE_HEAD_PATH),
                    "label_map_path": str(LABEL_MAP_PATH) if LABEL_MAP_PATH.exists() else None,
                }
            )
        else:
            logger.warning("Policy hot reload enabled but no initial checkpoint found")

    def _build_model(self, paths: Dict) -> Dict:
        policy_state = torch.load(paths.get("policy_head_path"), map_location=DEVICE)
        weight = policy_state.get("weight")
        if weight is None or not hasattr(weight, "shape"):
            raise ValueError("policy_head missing weight tensor")
        num_classes = int(weight.shape[0])

        label_map_path = Path(paths.get("label_map_path") or LABEL_MAP_PATH)
        label_map, source = _load_label_map_from_path(label_map_path if label_map_path else None)
        if not label_map and POLICY_LABEL_MAP_OPTIONAL:
            actions, source = _fallback_actions(num_classes)
            label_map = {action: idx for idx, action in enumerate(actions)}
            logger.warning("Label map missing; using fallback (%s) with %s actions", source, len(label_map))
        if not label_map:
            raise ValueError("label_map is empty")

        if num_classes != len(label_map):
            logger.warning(
                "Label map size (%s) does not match policy head classes (%s); reconciling",
                len(label_map),
                num_classes,
            )
        idx_to_action = _build_idx_to_action(label_map, num_classes)

        backbone = Backbone(frame_shape=FRAME_SHAPE, non_visual_dim=NON_VISUAL_DIM).to(DEVICE)
        policy_head = nn.Linear(backbone.output_dim, num_classes).to(DEVICE)
        value_head = nn.Linear(backbone.output_dim, 1).to(DEVICE)

        backbone_state = torch.load(paths.get("backbone_path"), map_location=DEVICE)
        backbone.load_state_dict(backbone_state)
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
            "label_map_source": source,
        }

    def _schedule_model_reload(self, paths: Dict):
        required = ["backbone_path", "policy_head_path"]
        if not all(paths.get(key) for key in required):
            logger.warning("Checkpoint missing required paths: %s", paths)
            return
        threading.Thread(target=self._reload_worker, args=(paths,), daemon=True).start()

    def _reload_worker(self, paths: Dict):
        if not self.reload_lock.acquire(blocking=False):
            logger.debug("Policy reload already in progress; skipping")
            return
        try:
            model = self._build_model(paths)
            with self.model_lock:
                self.model = model
            signature = self._checkpoint_signature()
            if signature is not None:
                self.last_checkpoint_signature = signature
            logger.info(
                "Policy model reloaded | actions=%s label_map=%s path=%s",
                len(model.get("idx_to_action", {})),
                model.get("label_map_source"),
                paths.get("policy_head_path"),
            )
        except Exception as exc:
            logger.error("Failed to load policy checkpoint: %s", exc)
        finally:
            self.reload_lock.release()

    def _get_model(self) -> Optional[Dict]:
        with self.model_lock:
            return self.model

    def _checkpoint_signature(self) -> Optional[Tuple[Tuple[float, int], ...]]:
        if not BACKBONE_PATH.exists() or not POLICY_HEAD_PATH.exists():
            return None
        paths = [BACKBONE_PATH, POLICY_HEAD_PATH]
        if LABEL_MAP_PATH.exists():
            paths.append(LABEL_MAP_PATH)
        signature = []
        for path in paths:
            stat = path.stat()
            signature.append((stat.st_mtime, stat.st_size))
        if VALUE_HEAD_PATH.exists():
            stat = VALUE_HEAD_PATH.stat()
            signature.append((stat.st_mtime, stat.st_size))
        return tuple(signature)

    def _checkpoint_paths_ready(self) -> bool:
        if self.auto_reload_delay_sec <= 0:
            return True
        now = time.time()
        paths = [BACKBONE_PATH, POLICY_HEAD_PATH]
        if LABEL_MAP_PATH.exists():
            paths.append(LABEL_MAP_PATH)
        if VALUE_HEAD_PATH.exists():
            paths.append(VALUE_HEAD_PATH)
        for path in paths:
            if not path.exists():
                continue
            stat = path.stat()
            if now - stat.st_mtime < self.auto_reload_delay_sec:
                return False
        return True

    def _auto_reload_loop(self) -> None:
        while True:
            time.sleep(self.auto_reload_sec)
            if not self.hot_reload_enabled or not self.auto_reload_enabled:
                continue
            try:
                self._maybe_auto_reload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Policy auto-reload check failed: %s", exc)

    def _maybe_auto_reload(self) -> None:
        if self.lazy_load_enabled and self.model is None:
            return
        signature = self._checkpoint_signature()
        if signature is None:
            return
        if not self._checkpoint_paths_ready():
            return
        if signature == self.last_checkpoint_signature:
            return
        self._reload_worker(
            {
                "backbone_path": str(BACKBONE_PATH),
                "policy_head_path": str(POLICY_HEAD_PATH),
                "value_head_path": str(VALUE_HEAD_PATH),
                "label_map_path": str(LABEL_MAP_PATH),
            }
        )

    # ------------------------------------------------------------------ MQTT
    def _on_connect(self, client, _userdata, _flags, rc):
        subscriptions = [(OBS_TOPIC, 0), (TEACHER_TOPIC, 0)]
        if SIM_TOPIC:
            subscriptions.append((SIM_TOPIC, 0))
        if GOAP_TOPIC:
            subscriptions.append((GOAP_TOPIC, 0))
        if CHECKPOINT_TOPIC:
            subscriptions.append((CHECKPOINT_TOPIC, 0))
        if PROGRESS_TOPIC:
            subscriptions.append((PROGRESS_TOPIC, 0))
        if REWARD_TOPIC:
            subscriptions.append((REWARD_TOPIC, 0))
        if GAME_SCHEMA_TOPIC:
            subscriptions.append((GAME_SCHEMA_TOPIC, 0))
        if CURSOR_TOPIC:
            subscriptions.append((CURSOR_TOPIC, 0))
        if MOTION_ANCHOR_TOPIC:
            subscriptions.append((MOTION_ANCHOR_TOPIC, 0))
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
        with self.state_lock:
            payload = msg.payload.decode("utf-8", "ignore")
            try:
                data = json.loads(payload)
            except Exception as exc:
                data = {}
                if payload.strip():
                    now = time.time()
                    if now - self.last_json_error_ts > 5.0:
                        logger.warning("Invalid JSON on %s: %s", msg.topic, exc)
                        self.last_json_error_ts = now

            if msg.topic == TEACHER_TOPIC:
                if isinstance(data, dict):
                    # Update game_id from teacher if not already set or changed
                    teacher_game_id = data.get("game_id") or data.get("context_game")
                    if teacher_game_id and teacher_game_id != self.game_id:
                        normalized = _normalize_game_id(str(teacher_game_id))
                        if normalized and normalized != "unknown_game":
                            self.game_id = normalized
                            logger.info("Policy game_id updated from teacher: %s", self.game_id)
                            self._maybe_refresh_profile_keys(self.game_id, "teacher_update")

                    raw_action = data.get("action") or data.get("text")
                    action_obj = raw_action if isinstance(raw_action, dict) else None
                    action_text = raw_action if isinstance(raw_action, str) else None
                    if action_obj and not action_text:
                        action_text = self._teacher_action_repr(action_obj)
                    if action_text:
                        self.teacher_action = {
                            "text": action_text,
                            "reasoning": data.get("reasoning"),
                            "timestamp": data.get("timestamp", time.time()),
                            "action": action_obj,
                        }
                        if action_obj:
                            if "target_norm" in action_obj and "target_norm" not in self.teacher_action:
                                self.teacher_action["target_norm"] = action_obj.get("target_norm")
                            if "target_label" in action_obj and "target_label" not in self.teacher_action:
                                self.teacher_action["target_label"] = action_obj.get("target_label")
                        if data.get("target_norm"):
                            self.teacher_action["target_norm"] = data.get("target_norm")
                        if data.get("target_label"):
                            self.teacher_action["target_label"] = data.get("target_label")
                        self._attach_teacher_target_hint(self.teacher_action)
                        logger.info("Received teacher action: %s", action_text)
            elif msg.topic == OBS_TOPIC or (SIM_TOPIC and msg.topic == SIM_TOPIC):
                self.latest_state = data
                self._update_game_id_from_scene(data)
                self._maybe_refresh_profile_keys(self.game_id, "control_profile")
                self._update_skill_feedback(data)
                tick_start = time.perf_counter()
                policy_action = self._policy_from_observation(data)
                if policy_action is None:
                    tick_ms = (time.perf_counter() - tick_start) * 1000.0
                    emit_latency(
                        client,
                        "policy",
                        tick_ms,
                        sla_ms=SLA_STAGE_POLICY_MS,
                        tags={"scene_ts": data.get("timestamp")},
                        agent="policy_agent",
                    )
                    self._record_control_metrics(client, tick_ms, None)
                    return
                final_action = self._blend_with_teacher(policy_action)
                if final_action is None:
                    tick_ms = (time.perf_counter() - tick_start) * 1000.0
                    emit_latency(
                        client,
                        "policy",
                        tick_ms,
                        sla_ms=SLA_STAGE_POLICY_MS,
                        tags={"scene_ts": data.get("timestamp")},
                        agent="policy_agent",
                    )
                    self._record_control_metrics(client, tick_ms, None)
                    return
                self._publish_action(client, final_action, policy_action)
                policy_ms = (time.perf_counter() - tick_start) * 1000.0
                emit_latency(
                    client,
                    "policy",
                    policy_ms,
                    sla_ms=SLA_STAGE_POLICY_MS,
                    tags={"scene_ts": data.get("timestamp")},
                    agent="policy_agent",
                )
                self._record_control_metrics(client, policy_ms, final_action)
            elif GOAP_TOPIC and msg.topic == GOAP_TOPIC:
                if data.get("status") == "pending":
                    self.task_queue.append(data)
                    logger.info(
                        "Queued GOAP task %s (%s) target=%s",
                        data.get("task_id"),
                        data.get("action_type"),
                        data.get("target"),
                    )
                    if not self.current_task:
                        self.current_task = self.task_queue[0]
            elif CHECKPOINT_TOPIC and msg.topic == CHECKPOINT_TOPIC and self.hot_reload_enabled:
                if isinstance(data, dict):
                    self._schedule_model_reload(data)
            elif PROGRESS_TOPIC and msg.topic == PROGRESS_TOPIC:
                if isinstance(data, dict):
                    self._handle_progress_status(data)
            elif REWARD_TOPIC and msg.topic == REWARD_TOPIC:
                if isinstance(data, dict):
                    self._handle_reward(data)
            elif GAME_SCHEMA_TOPIC and msg.topic == GAME_SCHEMA_TOPIC:
                schema = data.get("schema") if isinstance(data, dict) else None
                if not isinstance(schema, dict):
                    schema = data if isinstance(data, dict) else None
                if isinstance(schema, dict):
                    if POLICY_USE_UI_LAYOUT:
                        layout = schema.get("ui_layout") or {}
                        if isinstance(layout, dict):
                            play_area = layout.get("play_area")
                            hud = layout.get("hud_candidates")
                            if isinstance(play_area, dict):
                                self.play_area = play_area
                            if isinstance(hud, list):
                                self.hud_boxes = [box for box in hud if isinstance(box, dict)]
                    profile = schema.get("profile")
                    allowed = profile.get("allowed_keys") if isinstance(profile, dict) else None
                    if isinstance(allowed, list):
                        ordered = [str(k).lower() for k in allowed if k]
                        ordered = [k for k in ordered if self._skill_key_confirmed(k)]
                        keys = {k for k in ordered if k}
                        if keys:
                            merged = self.profile_allowed_keys | keys
                            if merged != self.profile_allowed_keys:
                                self.profile_allowed_keys = merged
                                logger.info("Policy merged allowed_keys from game schema: %s", sorted(merged))
                            self._maybe_refresh_profile_keys(self.game_id, "control_profile")
                            self._merge_profile_key_order(ordered, "game_schema")
            elif CURSOR_TOPIC and msg.topic == CURSOR_TOPIC:
                if isinstance(data, dict):
                    self._handle_cursor_message(data)
            elif MOTION_ANCHOR_TOPIC and msg.topic == MOTION_ANCHOR_TOPIC:
                if isinstance(data, dict):
                    self.motion_anchor = data

    def _update_game_id_from_scene(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        raw = state.get("game_id")
        if not raw:
            flags = state.get("flags") or {}
            raw = flags.get("game_id")
        if not raw:
            return
        normalized = _normalize_game_id(str(raw))
        if normalized and normalized != self.game_id:
            self.game_id = normalized
            logger.info("Policy game_id set: %s", self.game_id)

    def _load_profile_allowed_keys(self, game_id: str) -> Set[str]:
        ordered = self._load_profile_allowed_keys_ordered(game_id)
        return {k for k in ordered if k}

    def _load_profile_allowed_keys_ordered(self, game_id: str) -> List[str]:
        profile = load_profile(str(game_id)) or load_profile("unknown_game") or safe_profile(str(game_id))
        keys = [str(k).lower() for k in profile.get("allowed_keys", []) if k]
        if POLICY_INCLUDE_EXTENDED_KEYS:
            extended = [str(k).lower() for k in profile.get("allowed_keys_extended", []) if k]
            if POLICY_EXTENDED_KEYS_ALLOW:
                extended = [k for k in extended if k in POLICY_EXTENDED_KEYS_ALLOW]
            keys = keys + extended
        return keys

    def _merge_profile_key_order(self, keys: List[str], source: str) -> None:
        if not keys:
            return
        normalized = [str(k).lower() for k in keys if k]
        if not normalized:
            return
        merged = list(self.profile_allowed_key_order)
        existing = set(merged)
        for key in normalized:
            if key not in existing:
                merged.append(key)
                existing.add(key)
        if merged != self.profile_allowed_key_order:
            self.profile_allowed_key_order = merged
            logger.info("Policy merged allowed_keys order from %s: %s", source, merged)

    def _maybe_refresh_profile_keys(self, game_id: str, source: str) -> None:
        if not game_id:
            return
        ordered = self._load_profile_allowed_keys_ordered(game_id)
        keys = {k for k in ordered if k}
        if not keys:
            return
        merged = self.profile_allowed_keys | keys
        if merged != self.profile_allowed_keys:
            self.profile_allowed_keys = merged
            logger.info("Policy merged allowed_keys from %s: %s", source, sorted(merged))
        if ordered:
            self._merge_profile_key_order(ordered, source)

    def _attach_teacher_target_hint(self, payload: dict) -> None:
        if not payload:
            return
        reasoning = str(payload.get('reasoning') or '')
        text = str(payload.get('text') or '')
        existing_target_norm = payload.get('target_norm')
        existing_target_label = payload.get('target_label')
        if not reasoning and not text:
            if existing_target_norm is None:
                payload.pop('target_norm', None)
            if existing_target_label is None:
                payload.pop('target_label', None)
            return
        match = None
        for candidate in (reasoning, text):
            if not candidate:
                continue
            match = TEACHER_TARGET_COORD_RE.search(candidate)
            if match:
                break
            match = TEACHER_PAREN_COORD_RE.search(candidate)
            if match:
                break
        if match:
            try:
                x_norm = max(0.0, min(1.0, float(match.group(1))))
                y_norm = max(0.0, min(1.0, float(match.group(2))))
                payload['target_norm'] = [x_norm, y_norm]
            except ValueError:
                if existing_target_norm is None:
                    payload.pop('target_norm', None)
        else:
            if existing_target_norm is None:
                payload.pop('target_norm', None)
        hint_match = TEACHER_TARGET_LABEL_RE.search(reasoning) or TEACHER_TARGET_LABEL_RE.search(text)
        if hint_match:
            payload['target_label'] = hint_match.group(1).strip().strip('"')
        elif existing_target_label is None and 'target_label' in payload:
            payload.pop('target_label', None)
        target_label = payload.get('target_label')
        if target_label:
            resolved = self._resolve_scene_target_from_label(target_label)
            if resolved:
                payload['target_norm'] = resolved
        target_norm = payload.get('target_norm')
        if target_norm:
            self._set_active_target(target_norm, target_label)

    def _teacher_target_move(self, target_norm) -> Optional[Dict[str, object]]:
        if not target_norm:
            return None
        try:
            x_norm = float(target_norm[0])
            y_norm = float(target_norm[1])
        except (TypeError, ValueError, IndexError):
            return None
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        target_px, cursor_px, (delta_x, delta_y) = compute_cursor_motion(
            x_norm,
            y_norm,
            self.cursor_x_norm,
            self.cursor_y_norm,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            CURSOR_OFFSET_X,
            CURSOR_OFFSET_Y,
        )
        if abs(delta_x) <= 1 and abs(delta_y) <= 1:
            return None
        return {
            'label': 'mouse_move',
            'dx': int(delta_x),
            'dy': int(delta_y),
            'target_norm': [x_norm, y_norm],
            'target_px': target_px,
            'cursor_px': cursor_px,
            'source': 'teacher_target',
        }

    def _autoclick_recent(self, target_norm: List[float]) -> bool:
        if not self.last_autoclick:
            return False
        now = time.time()
        last_ts = self.last_autoclick.get("ts", 0.0)
        if (now - last_ts) > max(0.0, TEACHER_TARGET_AUTOCLICK_COOLDOWN):
            return False
        last_center = self.last_autoclick.get("center")
        if not last_center or len(last_center) != 2:
            return False
        try:
            dx = float(target_norm[0]) - float(last_center[0])
            dy = float(target_norm[1]) - float(last_center[1])
        except (TypeError, ValueError):
            return False
        dist = (dx * dx + dy * dy) ** 0.5
        return dist <= max(0.0, TEACHER_TARGET_AUTOCLICK_SAME_TOL)

    def _maybe_teacher_autoclick(
        self,
        target_norm: Optional[List[float]],
        target_label: Optional[str],
        text: str,
    ) -> Optional[Dict[str, object]]:
        if not TEACHER_TARGET_AUTOCLICK:
            return None
        if not target_norm or len(target_norm) != 2:
            return None
        if any(term in text for term in ("click", "attack", "shoot", "select", "press", "key")):
            return None
        try:
            x_norm = float(target_norm[0])
            y_norm = float(target_norm[1])
        except (TypeError, ValueError):
            return None
        if not self._cursor_is_fresh() or not self._cursor_near_target(x_norm, y_norm):
            return None
        if not self._click_ready():
            return None
        if self._autoclick_recent([x_norm, y_norm]):
            return None
        self.last_autoclick = {"center": [x_norm, y_norm], "ts": time.time()}
        payload = {"label": "click_primary", "target_norm": [x_norm, y_norm], "source": "teacher_autoclick"}
        if target_label:
            payload["target_label"] = target_label
        return payload


    def _set_active_target(self, center, label=None):
        if not center or len(center) != 2:
            self.active_target = None
            self.active_target_expires = 0.0
            return
        try:
            x_norm = max(0.0, min(1.0, float(center[0])))
            y_norm = max(0.0, min(1.0, float(center[1])))
        except (TypeError, ValueError):
            self.active_target = None
            self.active_target_expires = 0.0
            return
        self.active_target = {
            'center': [x_norm, y_norm],
            'label': label,
        }
        self.active_target_expires = time.time() + ACTIVE_TARGET_TTL

    def _update_scene_targets(self, state: dict) -> None:
        targets: List[Dict[str, object]] = []
        raw_targets = state.get('targets') if isinstance(state, dict) else None
        if isinstance(raw_targets, list):
            for entry in raw_targets:
                label = str(entry.get('label') or '').strip()
                center = entry.get('center')
                bbox = entry.get('bbox') or entry.get('box')
                if not label or not center or len(center) != 2:
                    continue
                try:
                    x_norm = max(0.0, min(1.0, float(center[0])))
                    y_norm = max(0.0, min(1.0, float(center[1])))
                except (TypeError, ValueError):
                    continue
                targets.append(
                    {
                        'label': label,
                        'norm_label': _normalize_phrase(label),
                        'center': [x_norm, y_norm],
                        'bbox': bbox,
                    }
                )
        elif isinstance(state.get('text_zones'), dict):
            for zone in state['text_zones'].values():
                label = str(zone.get('text') or '').strip()
                bbox = zone.get('bbox') or zone.get('box')
                if not label or not bbox or len(bbox) != 4:
                    continue
                try:
                    cx = max(0.0, min(1.0, float((bbox[0] + bbox[2]) / 2.0)))
                    cy = max(0.0, min(1.0, float((bbox[1] + bbox[3]) / 2.0)))
                except (TypeError, ValueError):
                    continue
                targets.append(
                    {
                        'label': label,
                        'norm_label': _normalize_phrase(label),
                        'center': [cx, cy],
                        'bbox': bbox,
                    }
                )
        self.scene_targets = targets
        if self.active_target and self.active_target.get('label'):
            refreshed = self._resolve_scene_target_from_label(self.active_target.get('label'))
            if refreshed:
                self.active_target['center'] = refreshed
                self.active_target_expires = time.time() + ACTIVE_TARGET_TTL
        player = state.get('player') if isinstance(state, dict) else None
        player_center = None
        if isinstance(player, dict):
            bbox = player.get('bbox')
            if bbox and len(bbox) == 4:
                try:
                    player_center = [float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0)]
                except Exception:
                    player_center = None
        if player_center:
            self.player_center = [max(0.0, min(1.0, player_center[0])), max(0.0, min(1.0, player_center[1]))]
        else:
            self.player_center = [0.5, 0.5]

    def _resolve_scene_target_from_label(self, label: str) -> Optional[List[float]]:
        normalized = _normalize_phrase(label)
        if not normalized:
            return None
        for entry in self.scene_targets:
            norm_label = entry.get('norm_label') or ''
            if not norm_label:
                continue
            if normalized in norm_label or norm_label in normalized:
                return entry.get('center')
        return None


    # ----------------------------------------------------------------- Policy
    def _policy_from_observation(self, data: dict) -> Optional[Dict[str, object]]:
        state = data or self.latest_state or {}
        if self.hot_reload_enabled and self.lazy_load_enabled and self.model is None:
            now = time.time()
            should_attempt = not self.model_load_attempted
            if not should_attempt and self.lazy_retry_sec > 0 and now >= self.lazy_next_retry_at:
                should_attempt = True
            if should_attempt:
                self.model_load_attempted = True
                try:
                    self._initial_model_load()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Policy lazy-load failed: %s", exc)
                if self.model is None and self.lazy_retry_sec > 0:
                    self.lazy_next_retry_at = now + self.lazy_retry_sec
                    if now - self.lazy_last_retry_log_ts >= self.lazy_retry_sec:
                        logger.warning(
                            "Policy lazy-load unavailable; retry scheduled in %.0fs",
                            self.lazy_retry_sec,
                        )
                        self.lazy_last_retry_log_ts = now
        self._update_scene_targets(state)
        if self.intent_enabled:
            self._update_intent(state)
        if self.active_target and time.time() > self.active_target_expires:
            self.active_target = None
        flags = state.get("flags") or {}
        if flags.get("death") and not self.respawn_macro_active:
            logger.info("Death flag detected; starting respawn macro immediately")
            self._start_respawn_macro()
        if self.respawn_macro_active:
            action = self._next_respawn_macro_action()
            if action is None:
                return {"label": "wait", "macro": "respawn"}
            return action
        if self.stage0_enabled:
            signature = self._state_signature(state)
            self.stage0_last_signature = signature
            if self.stage0_pending:
                if signature != self.stage0_reference:
                    self.stage0_pending = False
                elif time.time() < self.stage0_wait_until:
                    logger.debug(
                        "Stage0 waiting for settle | task=%s signature=%s ref=%s",
                        self.current_task.get("task_id") if self.current_task else None,
                        signature,
                        self.stage0_reference,
                    )
                    return None
                else:
                    logger.debug("Stage0: settle timeout reached, allowing next action")
                    self.stage0_pending = False
        allowed, block_reason = self._scene_allows_action(state)
        if not allowed:
            if block_reason != self.last_scene_block_reason:
                mean_val = state.get("mean", state.get("mean_brightness"))
                mean_note = ""
                try:
                    if mean_val is not None:
                        mean_note = f" mean={float(mean_val):.3f}"
                except (TypeError, ValueError):
                    mean_note = ""
                logger.warning(
                    "Policy pausing actions: scene not actionable (reason=%s%s)",
                    block_reason,
                    mean_note,
                )
            self.last_scene_block_reason = block_reason or "unknown"
            return None
        if self.last_scene_block_reason:
            logger.info("Policy resuming actions after scene block (%s)", self.last_scene_block_reason)
            self.last_scene_block_reason = None
        self._maybe_inject_respawn(state)
        if self.intent_enabled and self.current_intent in {"idle", "recover"}:
            if not (self.respawn_pending or self.respawn_macro_active):
                proactive = self._maybe_proactive_idle_action(state)
                if proactive:
                    return proactive
                return {"label": "wait", "reason": f"intent_{self.current_intent}"}
        try:
            mean_val = float(state.get("mean_brightness", state.get("mean", 0.0)))
        except (TypeError, ValueError):
            mean_val = 0.0
        signature = self._state_signature(state)
        if self.feedback_pending and signature == self.feedback_signature:
            logger.debug("Feedback pending settle; no state change yet")
            if FEEDBACK_SETTLE_SEC <= 0 or (time.time() - self.last_action_ts) < FEEDBACK_SETTLE_SEC:
                return None
        else:
            self.feedback_pending = False
        now = time.time()
        min_interval = STAGE0_ACTION_INTERVAL if self.stage0_enabled else DEBOUNCE
        if (now - self.last_action_ts) <= min_interval:
            logger.debug(
                "Stage0 debounce: last_action=%.2f now=%.2f interval=%.2f",
                self.last_action_ts,
                now,
                min_interval,
            )
            return None

        action = None
        if self.consecutive_moves > 30:
            logger.warning("Movement loop detected (%d moves); forcing target reset", self.consecutive_moves)
            self.active_target = None
            self.consecutive_moves = 0
            action = {"label": "wait", "reason": "movement_loop_break"}
        
        if not action:
            exploration_action = self._maybe_exploration_action(state)
            if exploration_action is not None:
                action = exploration_action
        
        if not action:
            if self.current_task:
                action = self._action_from_task(state)
            else:
                action = self._action_from_model(state)
                # ... (rest of model action selection)
            if (
                action
                and POLICY_PREFER_TARGETS
                and action.get("label") == "mouse_move"
            ):
                target_action = self._meaningful_fallback_action(state)
                if target_action and (target_action.get("target_norm") or target_action.get("target_label")):
                    action = target_action
            if self.intent_enabled and self.current_intent == "interact":
                interact_action = self._meaningful_fallback_action(state)
                if interact_action and (action is None or action.get("label") in {"mouse_move", "wait"}):
                    action = interact_action
            if action is None:
                if self.stage0_enabled:
                    return None
                
                # Secondary priority: Follow motion anchors
                action = self._motion_anchor_action(state)
                if action:
                    action["reason"] = "motion_anchor"
                
                if action is None:
                    action = self._meaningful_fallback_action(state)
                    if action:
                        action["reason"] = "meaningful_fallback"
                
                if action is None and POLICY_RANDOM_FALLBACK:
                    # Generic proactive fallback: if in combat intent or nothing else found, 
                    # do small moves/clicks around the center to stir things up
                    if self.rng.random() < 0.4:
                        # Click or move near center to stir things up
                        cx, cy = self.player_center if self.player_center else (0.5, 0.5)
                        rx = cx + self.rng.uniform(-0.15, 0.15)
                        ry = cy + self.rng.uniform(-0.15, 0.15)
                        label = "click_primary" if self.rng.random() < 0.5 else "mouse_move"
                        action = {"label": label, "target_norm": [rx, ry], "source": "proactive_idle", "reason": "proactive_idle"}
                    elif self.last_label == "mouse_move":
                        if self.rng.random() < 0.3:
                            dx = self._random_delta()
                            dy = self._random_delta()
                            action = {"label": "mouse_hold", "button": "left", "dx": dx, "dy": dy, "confidence": 1.0, "reason": "random_hold"}
                        else:
                            action = {"label": "click_primary", "confidence": min(1.0, mean_val / THRESH), "reason": "random_click"}
                    else:
                        if self.last_label == "mouse_hold":
                            action = {"label": "mouse_release", "button": "left", "confidence": 1.0, "reason": "random_release"}
                        else:
                            dx = self._random_delta()
                            dy = self._random_delta()
                            action = {"label": "mouse_move", "dx": dx, "dy": dy, "confidence": min(1.0, mean_val / THRESH), "reason": "random_move"}
                
                if action is None:
                    action = {"label": "wait", "reason": "final_fallback"}
        self.last_action_ts = now
        return action

    # ----------------------------------------------------------------- Scene
    def _scene_allows_action(self, state: dict) -> Tuple[bool, Optional[str]]:
        if self.respawn_macro_active:
            return True, None
        if not state:
            return False, "no_state"
        now = time.time()
        if POLICY_SCENE_MAX_AGE > 0:
            ts = state.get("timestamp") or state.get("scene_time") or state.get("scene_ts")
            try:
                if ts is not None and (now - float(ts)) > POLICY_SCENE_MAX_AGE:
                    return False, "stale_scene"
            except (TypeError, ValueError):
                pass
        if now < self.forbidden_until:
            return False, "forbidden_cooldown"
        flags = state.get("flags") or {}
        in_game = flags.get("in_game") is True
        if POLICY_REQUIRE_IN_GAME:
            if POLICY_REQUIRE_IN_GAME_STRICT:
                if not in_game:
                    return False, "not_in_game"
            elif flags.get("in_game") is False:
                return False, "not_in_game"
        if POLICY_REQUIRE_EMBEDDINGS and not self._embeddings_fresh(state):
            return False, "no_embeddings"
        if flags.get("death"):
            return True, None
        texts = self._collect_scene_texts(state)
        objects = state.get("objects") or []
        if in_game and self.desktop_keywords:
            texts = [entry for entry in texts if not self._text_list_matches([entry], self.desktop_keywords)]
        if self._scene_has_forbidden_ui(texts, objects):
            self.forbidden_until = now + max(0.0, POLICY_FORBIDDEN_COOLDOWN)
            return False, "forbidden_ui"
        if self.game_keywords:
            if in_game:
                return True, None
            has_keyword = self._scene_has_game_keyword(texts, objects)
            if has_keyword:
                return True, None
            mean_val = self._normalize_mean(state.get("mean", state.get("mean_brightness")))
            if mean_val is None:
                return False, "no_game_keywords"
            if mean_val >= POLICY_SCENE_BRIGHT_THRESHOLD:
                return False, "bright_scene_no_keywords"
            return True, None
        if not in_game and self.desktop_keywords and self._text_list_matches(texts, self.desktop_keywords):
            self.forbidden_until = now + max(DESKTOP_PAUSE_SEC, POLICY_FORBIDDEN_COOLDOWN)
            return False, "desktop_detected"
        return True, None

    def _maybe_inject_respawn(self, state: dict) -> bool:
        if self.respawn_pending:
            return False
        
        flags = state.get("flags") or {}
        now = time.time()
        
        # Priority 1: Direct death flag from scene agent
        is_dead = flags.get("death")
        
        # Priority 2: OCR text match
        if not is_dead:
            texts = self._collect_scene_texts(state)
            if texts and self._scene_has_respawn_text(texts):
                is_dead = True
        
        if not is_dead:
            return False
            
        if (now - self.last_respawn_ts) < RESPAWN_COOLDOWN:
            return False
            
        self._queue_respawn_tasks()
        self.last_respawn_ts = now
        logger.info("Auto-respawn triggered (flag=%s) -> queueing WAIT+MOVE+CLICK", flags.get("death"))
        return True

    def _scene_has_respawn_text(self, texts) -> bool:
        for entry in texts:
            raw = str(entry)
            if not raw:
                continue
            cleaned = _normalize_phrase(raw)
            if self._matches_respawn_keyword(cleaned):
                logger.info("Detected respawn keyword in text='%s' cleaned='%s'", raw[:80], cleaned)
                return True
        return False

    def _collect_scene_texts(self, state: Optional[dict]) -> List[str]:
        if not isinstance(state, dict):
            return []
        return _gather_scene_texts(state)

    def _matches_respawn_keyword(self, cleaned: str) -> bool:
        if not cleaned:
            return False
        for candidate in _candidate_chunks(cleaned):
            for keyword in self.respawn_keywords:
                if not keyword:
                    continue
                ratio = difflib.SequenceMatcher(None, keyword, candidate).ratio()
                if RESPAWN_DEBUG:
                    logger.info(
                        "Respawn fuzzy check keyword='%s' candidate='%s' ratio=%.3f",
                        keyword,
                        candidate,
                        ratio,
                    )
                if ratio >= RESPAWN_FUZZY_THRESHOLD:
                    return True
                if RESPAWN_SKELETON_THRESHOLD > 0:
                    skeleton_keyword = _normalize_skeleton(keyword)
                    skeleton_candidate = _normalize_skeleton(candidate)
                    if (
                        len(skeleton_keyword) >= RESPAWN_SKELETON_MIN_LEN
                        and len(skeleton_candidate) >= RESPAWN_SKELETON_MIN_LEN
                    ):
                        skeleton_ratio = difflib.SequenceMatcher(
                            None,
                            skeleton_keyword,
                            skeleton_candidate,
                        ).ratio()
                        if RESPAWN_DEBUG:
                            logger.info(
                                "Respawn skeleton check keyword='%s' candidate='%s' ratio=%.3f",
                                skeleton_keyword,
                                skeleton_candidate,
                                skeleton_ratio,
                            )
                        if skeleton_ratio >= RESPAWN_SKELETON_THRESHOLD:
                            return True
        return False

    def _queue_respawn_tasks(self):
        goal_id = f"respawn_{int(time.time())}"
        target = {
            "x_norm": RESPAWN_TARGET_X,
            "y_norm": RESPAWN_TARGET_Y,
            "x": RESPAWN_TARGET_X,
            "y": RESPAWN_TARGET_Y,
            "button": "primary",
            "area": "critical_dialog",
            "scope": "critical_dialog:death",
            "target_source": "auto_respawn",
        }
        wait_task = {
            "goal_id": goal_id,
            "task_id": f"respawn_wait_{int(time.time() * 1000)}",
            "action_type": "WAIT",
            "duration": 1.5,
            "status": "pending",
        }
        move_task = {
            "goal_id": goal_id,
            "task_id": f"respawn_move_{int(time.time() * 1000)}",
            "action_type": "MOVE_TO",
            "target": dict(target, target_norm=[RESPAWN_TARGET_X, RESPAWN_TARGET_Y]),
            "status": "pending",
        }
        click_task = {
            "goal_id": goal_id,
            "task_id": f"respawn_click_{int(time.time() * 1000)}",
            "action_type": "CLICK_BUTTON",
            "target": dict(target, target_norm=[RESPAWN_TARGET_X, RESPAWN_TARGET_Y]),
            "status": "pending",
        }
        self.task_queue.clear()
        self.task_queue.extend([wait_task, move_task, click_task])
        self.current_task = self.task_queue[0]
        self.respawn_pending = True

    def _text_has_game_keyword(self, text: str) -> bool:
        if not text or not self.game_keywords:
            return False
        lowered = text.lower()
        compact = "".join(ch for ch in lowered if ch.isalnum())
        return any(keyword in lowered or keyword in compact for keyword in self.game_keywords)

    def _scene_has_game_keyword(self, texts, objects) -> bool:
        if not self.game_keywords:
            return False
        for entry in texts:
            if self._text_has_game_keyword(str(entry)):
                return True
        for obj in objects:
            descriptor = " ".join(
                str(obj.get(field) or "")
                for field in ("label", "text", "class", "name")
            )
            if descriptor and self._text_has_game_keyword(descriptor):
                return True
        return False

    @staticmethod
    def _text_list_matches(entries, keywords) -> bool:
        if not keywords:
            return False
        for entry in entries:
            cleaned = _normalize_phrase(str(entry))
            if not cleaned:
                continue
            for keyword in keywords:
                if keyword and keyword in cleaned:
                    return True
        return False

    def _handle_cursor_message(self, payload: Dict[str, object]):
        self.latest_cursor = payload
        if payload.get("ok"):
            x = payload.get("x_norm")
            y = payload.get("y_norm")
            if x is None or y is None:
                return
            try:
                self.cursor_x_norm = float(max(0.0, min(1.0, x)))
                self.cursor_y_norm = float(max(0.0, min(1.0, y)))
                self.cursor_detected_ts = time.time()
            except (TypeError, ValueError):
                return
        else:
            self.cursor_detected_ts = 0.0

    def _handle_progress_status(self, payload: Dict[str, object]) -> None:
        understanding = payload.get("understanding") or {}
        locations = understanding.get("locations") or {}
        objects = understanding.get("objects") or {}
        last_reward_age = payload.get("last_reward_age")
        try:
            last_reward_age = float(last_reward_age) if last_reward_age is not None else None
        except (TypeError, ValueError):
            last_reward_age = None
        self.progress_state = {
            "last_reward_age": last_reward_age,
            "locations_current_age_sec": locations.get("current_age_sec"),
            "locations_new_rate": locations.get("new_rate"),
            "objects_new_rate": objects.get("new_rate"),
            "objects_vocab_size": objects.get("vocab_size"),
            "timestamp": payload.get("timestamp", time.time()),
        }

    def _handle_reward(self, payload: Dict[str, object]) -> None:
        reward = payload.get("reward")
        try:
            reward_val = float(reward)
        except (TypeError, ValueError):
            return
        if abs(reward_val) >= POLICY_EXPLORATION_MIN_REWARD:
            self.last_reward_ts = time.time()

    def _cursor_is_fresh(self) -> bool:
        if not POLICY_USE_CURSOR:
            return False
        return self.cursor_detected_ts > 0 and (time.time() - self.cursor_detected_ts) <= CURSOR_TIMEOUT_SEC

    def _cursor_near_target(self, x_norm: float, y_norm: float) -> bool:
        if not self._cursor_is_fresh():
            # If cursor is lost, we assume it's roughly at the last target 
            # to allow blind clicks and prevent getting stuck in move loops
            return True
        return (
            abs(self.cursor_x_norm - x_norm) <= CURSOR_TOLERANCE
            and abs(self.cursor_y_norm - y_norm) <= CURSOR_TOLERANCE
        )

    def _embeddings_fresh(self, state: dict) -> bool:
        if not EMBED_FEATURE_ENABLED:
            return False
        embedding = state.get("embeddings") or state.get("embedding")
        if not isinstance(embedding, (list, tuple)) or not embedding:
            return False
        if EMBED_FEATURE_SOURCE_DIM and len(embedding) < EMBED_FEATURE_SOURCE_DIM:
            return False
        ts = state.get("embeddings_ts") or state.get("embedding_ts")
        if ts is None or POLICY_EMBED_MAX_AGE_SEC <= 0:
            return True
        try:
            return (time.time() - float(ts)) <= POLICY_EMBED_MAX_AGE_SEC
        except (TypeError, ValueError):
            return True

    def _reward_age_sec(self) -> Optional[float]:
        age = self.progress_state.get("last_reward_age")
        if isinstance(age, (int, float)):
            return float(age)
        if self.last_reward_ts:
            return time.time() - self.last_reward_ts
        return None

    def _actions_repeating(self) -> bool:
        if len(self.recent_action_labels) < POLICY_EXPLORATION_REPEAT_WINDOW:
            return False
        counts = Counter(self.recent_action_labels)
        min_needed = min(POLICY_EXPLORATION_REPEAT_MIN, POLICY_EXPLORATION_REPEAT_WINDOW)
        return max(counts.values()) >= min_needed

    def _exploration_targets(self) -> List[Tuple[float, float]]:
        margin = max(0.01, min(0.3, POLICY_EXPLORATION_MARGIN))
        return [
            (margin, margin),
            (1.0 - margin, margin),
            (margin, 1.0 - margin),
            (1.0 - margin, 1.0 - margin),
            (0.5, margin),
            (0.5, 1.0 - margin),
            (margin, 0.5),
            (1.0 - margin, 0.5),
            (0.5, 0.5),
        ]

    def _exploration_move_action(self, x_norm: float, y_norm: float) -> Dict[str, object]:
        target_px, cursor_px, (delta_x, delta_y) = compute_cursor_motion(
            x_norm,
            y_norm,
            self.cursor_x_norm,
            self.cursor_y_norm,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            CURSOR_OFFSET_X,
            CURSOR_OFFSET_Y,
        )
        if abs(delta_x) < MIN_MOUSE_DELTA and abs(delta_y) < MIN_MOUSE_DELTA:
            delta_x = self.rng.randint(-POLICY_EXPLORATION_MOUSE_RANGE, POLICY_EXPLORATION_MOUSE_RANGE)
            delta_y = self.rng.randint(-POLICY_EXPLORATION_MOUSE_RANGE, POLICY_EXPLORATION_MOUSE_RANGE)
        return {
            "label": "mouse_move",
            "dx": int(delta_x),
            "dy": int(delta_y),
            "target_norm": [x_norm, y_norm],
            "target_px": target_px,
            "cursor_px": cursor_px,
            "source": "exploration",
        }

    def _queue_exploration_burst(self) -> None:
        now = time.time()
        if now < self.exploration_cooldown_until:
            return
        targets = self._exploration_targets()
        self.rng.shuffle(targets)
        actions: List[Dict[str, object]] = []
        last_target = None
        for target in targets:
            if len(actions) >= max(1, POLICY_EXPLORATION_BURST_ACTIONS):
                break
            actions.append(self._exploration_move_action(target[0], target[1]))
            last_target = target
        if POLICY_EXPLORATION_ALLOW_CLICK and actions:
            click_action = {"label": "click_primary", "source": "exploration"}
            if last_target is not None:
                click_action["target_norm"] = [last_target[0], last_target[1]]
            actions.append(click_action)
        explore_keys = POLICY_EXPLORATION_KEYS
        if not explore_keys and POLICY_EXPLORATION_KEYS_FROM_PROFILE:
            explore_keys = self.profile_allowed_keys
        if explore_keys:
            keys_sorted = sorted(explore_keys)
            if POLICY_EXPLORATION_KEY_BURST and now >= self.exploration_key_cooldown_until:
                self.rng.shuffle(keys_sorted)
                count = max(1, POLICY_EXPLORATION_KEY_BURST_COUNT)
                for key in keys_sorted[:count]:
                    actions.append({"label": "key_press", "key": key, "source": "exploration"})
                self.exploration_key_cooldown_until = now + max(0.0, POLICY_EXPLORATION_KEY_BURST_COOLDOWN_SEC)
            elif self.rng.random() < 0.3:
                key = self.rng.choice(keys_sorted)
                actions.append({"label": "key_press", "key": key, "source": "exploration"})
        if actions:
            self.exploration_queue.extend(actions)
            self.exploration_cooldown_until = now + POLICY_EXPLORATION_COOLDOWN_SEC
            logger.info(
                "Exploration burst queued | actions=%s reward_age=%.1fs loc_age=%s obj_new_rate=%s",
                len(actions),
                self._reward_age_sec() or -1,
                self.progress_state.get("locations_current_age_sec"),
                self.progress_state.get("objects_new_rate"),
            )

    def _maybe_exploration_action(self, state: dict) -> Optional[Dict[str, object]]:
        if self.exploration_queue:
            return self.exploration_queue.popleft()
        if not POLICY_EXPLORATION_ENABLED:
            return None
        if self.intent_enabled and self.current_intent in {"combat", "recover"}:
            return None
        if self.current_task or self.respawn_macro_active or self.respawn_pending:
            return None
        if self._scene_is_forbidden():
            return None
        reward_age = self._reward_age_sec()
        if reward_age is None or reward_age < POLICY_EXPLORATION_REWARD_TIMEOUT_SEC:
            return None
        loc_age = self.progress_state.get("locations_current_age_sec")
        if isinstance(loc_age, (int, float)) and float(loc_age) < POLICY_EXPLORATION_LOCATION_AGE_SEC:
            return None
        obj_rate = self.progress_state.get("objects_new_rate")
        if isinstance(obj_rate, (int, float)) and float(obj_rate) > POLICY_EXPLORATION_OBJECT_NEW_RATE_MAX:
            return None
        if not self._actions_repeating():
            return None
        self._queue_exploration_burst()
        if self.exploration_queue:
            return self.exploration_queue.popleft()
        return None

    def _maybe_proactive_idle_action(self, state: dict) -> Optional[Dict[str, object]]:
        if PROACTIVE_IDLE_CHANCE <= 0:
            return None
        flags = state.get("flags") or {}
        if flags.get("in_game") is False:
            return None
        if self.rng.random() > PROACTIVE_IDLE_CHANCE:
            return None
        cx, cy = self.player_center if self.player_center else (0.5, 0.5)
        rx = max(0.0, min(1.0, cx + self.rng.uniform(-PROACTIVE_IDLE_RADIUS, PROACTIVE_IDLE_RADIUS)))
        ry = max(0.0, min(1.0, cy + self.rng.uniform(-PROACTIVE_IDLE_RADIUS, PROACTIVE_IDLE_RADIUS)))
        move_prob = max(0.0, min(1.0, PROACTIVE_IDLE_MOVE_PROB))
        if self.rng.random() >= move_prob:
            return {"label": "click_primary", "target_norm": [rx, ry], "source": "proactive_idle"}
        dx = int((rx - self.cursor_x_norm) * SCREEN_WIDTH)
        dy = int((ry - self.cursor_y_norm) * SCREEN_HEIGHT)
        if dx == 0 and dy == 0:
            dx = self._random_delta()
            dy = self._random_delta()
        return {"label": "mouse_move", "dx": dx, "dy": dy, "target_norm": [rx, ry], "source": "proactive_idle"}

    @staticmethod
    def _center_from_bbox(bbox: Optional[List[float]]) -> Optional[List[float]]:
        if not bbox or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        if max(x1, y1, x2, y2) > 1.5:
            return None
        cx = max(0.0, min(1.0, (x1 + x2) / 2.0))
        cy = max(0.0, min(1.0, (y1 + y2) / 2.0))
        return [cx, cy]

    def _center_in_box(self, center: Tuple[float, float], box: Dict[str, float]) -> bool:
        try:
            x, y = float(center[0]), float(center[1])
            bx = float(box.get("x", 0.0))
            by = float(box.get("y", 0.0))
            bw = float(box.get("w", 0.0))
            bh = float(box.get("h", 0.0))
        except (TypeError, ValueError):
            return False
        return bx <= x <= (bx + bw) and by <= y <= (by + bh)

    def _center_in_hud(self, center: Tuple[float, float]) -> bool:
        if not self.hud_boxes:
            return False
        for box in self.hud_boxes:
            if self._center_in_box(center, box):
                return True
        return False

    def _pick_object_target(self, objects: List[dict]) -> Optional[Tuple[List[float], Optional[str]]]:
        best_center = None
        best_label = None
        best_score = -1.0
        for obj in objects:
            center = obj.get("center") or self._center_from_bbox(obj.get("bbox") or obj.get("box"))
            if not center:
                continue
            try:
                score = float(obj.get("confidence") or obj.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score > best_score:
                best_score = score
                best_center = center
                best_label = str(obj.get("label") or obj.get("class") or "")
        if not best_center:
            return None
        return best_center, best_label

    def _pick_enemy_target(self, enemies: List[dict]) -> Optional[List[float]]:
        if not enemies:
            return None
        px, py = self.player_center if self.player_center else (0.5, 0.5)
        centers = []
        for obj in enemies:
            center = obj.get("center") or self._center_from_bbox(obj.get("bbox") or obj.get("box"))
            if not center:
                continue
            try:
                cx, cy = float(center[0]), float(center[1])
            except (TypeError, ValueError, IndexError):
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            if POLICY_USE_UI_LAYOUT and self._center_in_hud((cx, cy)):
                continue
            if POLICY_USE_UI_LAYOUT and self.play_area and not self._center_in_box((cx, cy), self.play_area):
                continue
            centers.append((cx, cy))
        if not centers:
            return None
        if len(centers) >= max(2, POLICY_ENEMY_CLUSTER_MIN):
            avg_x = sum(c[0] for c in centers) / len(centers)
            avg_y = sum(c[1] for c in centers) / len(centers)
            return [avg_x, avg_y]
        best_center = None
        best_dist = 1e9
        for cx, cy in centers:
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < best_dist:
                best_dist = dist
                best_center = [cx, cy]
        return best_center

    def _enemy_bar_score(self, state: dict) -> Optional[float]:
        bars = state.get("enemy_bars")
        if isinstance(bars, list) and bars:
            score = 0.0
            for bar in bars:
                bbox = bar.get("bbox") or bar.get("box")
                if not bbox or len(bbox) != 4:
                    continue
                try:
                    score += max(0.0, float(bbox[2]) - float(bbox[0]))
                except (TypeError, ValueError):
                    continue
            return score
        enemies = state.get("enemies")
        if isinstance(enemies, list) and enemies:
            return float(len(enemies))
        return None

    def _update_skill_feedback(self, state: dict) -> None:
        if not self.last_combat_key:
            return
        if (time.time() - self.last_combat_ts) < POLICY_SKILL_FEEDBACK_SEC:
            return
        score = self._enemy_bar_score(state)
        if score is None or self.last_combat_score is None:
            self.last_combat_key = None
            return
        delta = self.last_combat_score - score
        if delta < POLICY_ENEMY_BAR_MIN_DELTA:
            self.last_combat_key = None
            return
        stats = self.skill_stats.setdefault(self.last_combat_key, {"hits": 0.0, "tries": 0.0})
        stats["hits"] += 1.0
        stats["tries"] += 1.0
        self.last_combat_key = None

    def _record_skill_try(self, key: str, state: dict) -> None:
        stats = self.skill_stats.setdefault(key, {"hits": 0.0, "tries": 0.0})
        stats["tries"] += 1.0
        self.last_combat_key = key
        self.last_combat_ts = time.time()
        self.last_combat_score = self._enemy_bar_score(state)

    def _choose_skill_key(self) -> Optional[str]:
        skill_set = set(POLICY_SKILL_KEYS)
        candidates = [k for k in POLICY_SKILL_KEYS if (not self.profile_allowed_keys or k in self.profile_allowed_keys)]
        if self.profile_allowed_key_order:
            ordered = [k for k in self.profile_allowed_key_order if k in skill_set]
            if ordered:
                candidates = ordered
        
        # If no candidates from profile, use fallbacks from env
        if not candidates and POLICY_FALLBACK_SKILL_KEYS:
            candidates = list(POLICY_FALLBACK_SKILL_KEYS)
        
        # Final fallback to common gaming keys if everything else fails but enemies are present
        if not candidates and self._enemies_present():
            # Universal set: Numbers, common action keys
            candidates = ["1", "2", "3", "4", "5", "q", "e", "r", "f", "space"]

        if not candidates:
            return None

        candidates = [k for k in candidates if self._skill_key_confirmed(k)]
        if not candidates:
            return None
        
        # epsilon-greedy based on observed hit rate
        if self.rng.random() < max(0.0, min(1.0, POLICY_SKILL_EPS)):
            return self.rng.choice(candidates)
        best_key = None
        best_score = -1.0
        best_keys = []
        for key in candidates:
            stats = self.skill_stats.get(key)
            if not stats or stats.get("tries", 0.0) <= 0:
                score = 0.0
            else:
                score = stats.get("hits", 0.0) / max(1.0, stats.get("tries", 1.0))
            if score > best_score:
                best_score = score
                best_keys = [key]
                best_key = key
            elif score == best_score:
                best_keys.append(key)
        if best_keys:
            return self.rng.choice(best_keys)
        return best_key or candidates[0]

    def _pick_text_target(self, targets: List[dict]) -> Optional[Tuple[List[float], Optional[str]]]:
        best_center = None
        best_label = None
        best_score = -1.0
        px, py = self.player_center if self.player_center else (0.5, 0.5)
        
        # UI ignore list for targeting
        ui_ignore = {"quest", "objective", "mana", "life", "level", "xp", "energy", "gate"}
        
        for target in targets:
            label = str(target.get("label") or "").strip()
            lowered_label = label.lower()
            
            # Filter 1: Keywords
            if any(u in lowered_label for u in ui_ignore):
                continue
            
            # Filter 2: Too long or ALL CAPS headers
            if len(label) > 25 or (label.isupper() and len(label) > 4):
                continue
                
            center = target.get("center") or self._center_from_bbox(target.get("bbox"))
            if not label or not center:
                continue
            if self._text_hits_forbidden(label):
                continue
            try:
                cx, cy = float(center[0]), float(center[1])
            except (TypeError, ValueError, IndexError):
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            if POLICY_USE_UI_LAYOUT and self._center_in_hud((cx, cy)):
                continue
            if POLICY_USE_UI_LAYOUT and self.play_area and not self._center_in_box((cx, cy), self.play_area):
                continue
            dist = (cx - px) ** 2 + (cy - py) ** 2
            length_bonus = min(1.0, len(label) / 20.0) * 0.15
            score = (1.0 - dist) + length_bonus
            if score > best_score:
                best_score = score
                best_center = [cx, cy]
                best_label = label
        if not best_center:
            return None
        return best_center, best_label

    def _move_or_click_target(self, center: List[float], label: Optional[str]) -> Dict[str, object]:
        try:
            x_norm = float(center[0])
            y_norm = float(center[1])
        except (TypeError, ValueError, IndexError):
            return {"label": "wait"}
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        if self._cursor_near_target(x_norm, y_norm):
            # If we are already there, we MUST try to interact, otherwise we just sit there
            if self._click_ready():
                payload = {"label": "click_primary", "target_norm": [x_norm, y_norm]}
                if label:
                    payload["target_label"] = label
                return payload
            else:
                # If click is on cooldown but we are at target, don't just wait, 
                # maybe use a skill or do something else to avoid being a sitting duck
                if self._enemies_present() and self._enemy_skill_ready():
                    skill_key = self._choose_skill_key()
                    if skill_key:
                        return {"label": "key_press", "key": skill_key, "source": "combat_at_target"}
                return {"label": "wait"}
        
        move = self._teacher_target_move([x_norm, y_norm])
        if move:
            # If the move is very small, treat it as 'arrived' to trigger a click next tick
            dx = move.get("dx", 0)
            dy = move.get("dy", 0)
            if abs(dx) < 5 and abs(dy) < 5 and self.rng.random() < 0.5:
                 if self._click_ready():
                    return {"label": "click_primary", "target_norm": [x_norm, y_norm]}
            return move
        return {"label": "wait"}

    def _motion_anchor_action(self, state: Optional[dict] = None) -> Optional[Dict[str, object]]:
        if not self.motion_anchor:
            return None
        ts = self.motion_anchor.get("ts") or self.motion_anchor.get("timestamp")
        try:
            if not ts or (time.time() - float(ts)) > MOTION_ANCHOR_TTL:
                return None
        except (TypeError, ValueError):
            return None

        point = self.motion_anchor.get("point") or self.motion_anchor.get("center")
        if not point or len(point) != 2:
            return None

        # Return a move toward the salient motion point
        move = self._teacher_target_move(point)
        if move:
            move["label"] = "mouse_move"
            move["source"] = "motion_anchor"
        return move

    def _meaningful_fallback_action(self, state: dict) -> Optional[Dict[str, object]]:
        if not POLICY_MEANINGFUL_FALLBACK:
            return None
        if POLICY_REQUIRE_EMBEDDINGS and not self._embeddings_fresh(state):
            return None

        if self.active_target and time.time() <= self.active_target_expires:
            center = self.active_target.get("center") or self.active_target.get("target_norm")
            if center:
                return self._move_or_click_target(center, self.active_target.get("label"))

        enemies = state.get("enemies") or []
        target = self._pick_object_target(enemies)
        if target:
            center, label = target
            return self._move_or_click_target(center, label)

        # Reduce OCR target frequency - don't chase text 100% of the time
        if POLICY_USE_OCR_TARGETS and self.rng.random() < 0.3:
            targets = state.get("targets") or []
            target = self._pick_text_target(targets)
            if target:
                center, label = target
                return self._move_or_click_target(center, label)

        objects = state.get("objects") or []
        target = self._pick_object_target(objects)
        if target:
            center, label = target
            return self._move_or_click_target(center, label)
        return None

    def _hover_confirms_target(self, task: Dict[str, object], target: Dict[str, object]) -> bool:
        if not HOVER_VERIFY_ENABLED:
            if HOVER_DEBUG:
                logger.debug("Hover skip: no text hint task=%s target=%s", task.get("task_id"), target)
            return True
        text_hint = target.get("text") or target.get("text_hint")
        if not text_hint:
            self.hover_failures.pop(self._hover_failure_key(task, target), None)
            if HOVER_DEBUG:
                logger.debug("Hover skip: empty reference task=%s", task.get("task_id"))
            return True
        reference = _normalize_phrase(str(text_hint))
        if not reference:
            self.hover_failures.pop(self._hover_failure_key(task, target), None)
            return True
        state = self.latest_state or {}
        entries = self._collect_scene_texts(state)
        for entry in entries:
            cleaned = _normalize_phrase(str(entry))
            if not cleaned:
                continue
            if SHOP_HOVER_BLOCK_ENABLED and self._text_hits_forbidden(str(entry)):
                logger.info(
                    "Hover forbidden text entry=%s task=%s",
                    entry[:80],
                    task.get("task_id"),
                )
                return False
            ratio = difflib.SequenceMatcher(None, reference, cleaned).ratio()
            if ratio >= HOVER_FUZZY_THRESHOLD:
                logger.info(
                    "Hover matched reference=%s entry=%s ratio=%.3f task=%s",
                    reference,
                    entry[:80],
                    ratio,
                    task.get("task_id"),
                )
                self.hover_failures.pop(self._hover_failure_key(task, target), None)
                return True
            else:
                if HOVER_DEBUG:
                    logger.debug(
                        "Hover mismatch ref=%s entry=%s ratio=%.3f task=%s",
                        reference,
                        entry[:80],
                        ratio,
                        task.get("task_id"),
                    )
        if self._should_fallback_hover(task, target):
            return True
        if HOVER_DEBUG:
            logger.debug(
                "Hover failed ref=%s task=%s cursor=(%.3f,%.3f) area=%s",
                text_hint,
                task.get("task_id"),
                self.cursor_x_norm,
                self.cursor_y_norm,
                self.latest_cursor.get("area"),
            )
        return False

    def _hover_hits_forbidden_text(self) -> bool:
        if not SHOP_HOVER_BLOCK_ENABLED or not self._cursor_is_fresh():
            return False
        state = self.latest_state or {}
        entries = self._collect_scene_texts(state)
        return any(self._text_hits_forbidden(str(entry)) for entry in entries)

    def _hover_failure_key(self, task: Dict[str, object], target: Dict[str, object]) -> str:
        return str(
            task.get("task_id")
            or target.get("target_id")
            or target.get("text")
            or target.get("text_hint")
            or target.get("area")
            or "hover"
        )

    def _should_fallback_hover(self, task: Dict[str, object], target: Dict[str, object]) -> bool:
        if HOVER_FAIL_LIMIT <= 0:
            return False
        key = self._hover_failure_key(task, target)
        now = time.time()
        entry = self.hover_failures.get(key)
        if entry and now - entry.get("first_ts", now) > HOVER_FAIL_WINDOW:
            entry = None
        if entry is None:
            self.hover_failures[key] = {"count": 1, "first_ts": now}
            return False
        entry["count"] = int(entry.get("count", 0)) + 1
        if entry["count"] >= HOVER_FAIL_LIMIT:
            self.hover_failures.pop(key, None)
            logger.warning(
                "Hover confirmation fallback after %s fails task=%s target_hint=%s",
                entry["count"],
                task.get("task_id"),
                target.get("text") or target.get("text_hint"),
            )
            return True
        return False

    def _text_hits_forbidden(self, text: str) -> bool:
        if not text or not self.forbidden_texts:
            return False
        lowered = text.lower()
        compact = "".join(ch for ch in lowered if ch.isalnum())
        return any(token in lowered or token in compact for token in self.forbidden_texts)

    def _scene_is_forbidden(self) -> bool:
        state = self.latest_state or {}
        texts = self._collect_scene_texts(state)
        objects = state.get("objects") or []
        return self._scene_has_forbidden_ui(texts, objects)

    def _normalize_mean(self, raw_value) -> Optional[float]:
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        if value > 2.0:
            value /= 255.0
        return max(0.0, min(1.0, value))

    def _scene_has_forbidden_ui(self, texts, objects) -> bool:
        if not self.forbidden_texts:
            return False
        for entry in texts:
            if self._text_hits_forbidden(str(entry)):
                return True
        for obj in objects:
            for field in ("label", "text", "class", "name", "title"):
                if self._text_hits_forbidden(str(obj.get(field) or "")):
                    return True
            tags = obj.get("tags") or []
            normalized_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
            if normalized_tags & self.forbidden_tags:
                return True
        return False

    def _latest_scene_has_forbidden(self) -> bool:
        state = self.latest_state or {}
        texts = self._collect_scene_texts(state)
        objects = state.get("objects") or []
        return self._scene_has_forbidden_ui(texts, objects)

    def _latest_scene_has_respawn_text(self) -> bool:
        state = self.latest_state or {}
        texts = self._collect_scene_texts(state)
        return self._scene_has_respawn_text(texts)

    def _should_suppress_shop(self) -> bool:
        if not self.shop_suppress_enabled:
            return False
        if not self._latest_scene_has_forbidden():
            return False
        if not self.shop_suppress_death_only:
            return True
        state = self.latest_state or {}
        flags = state.get("flags") or {}
        if flags.get("death"):
            return True
        if self._latest_scene_has_respawn_text():
            return True
        if self.respawn_pending:
            return True
        return False

    def _is_forbidden_target(self, target: Optional[dict]) -> bool:
        if not target:
            return False
        tags = target.get("tags") or []
        normalized_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        if self.forbidden_tags and normalized_tags & self.forbidden_tags:
            return True
        for field in ("text", "label", "name", "title"):
            if self._text_hits_forbidden(str(target.get(field) or "")):
                return True
        return False

    def _current_alpha(self) -> float:
        progress = min(self.steps / float(self.teacher_decay_steps), 1.0)
        alpha = self.teacher_alpha_start * (1.0 - progress)
        return max(self.teacher_min_alpha, min(1.0, alpha))

    @staticmethod
    def _is_respawn_task(task: Optional[Dict[str, object]]) -> bool:
        if not isinstance(task, dict):
            return False
        for field in ("task_id", "goal_id", "action_type"):
            value = str(task.get(field) or "").lower()
            if "respawn" in value:
                return True
        return False

    def _blend_with_teacher(self, policy_action: Dict[str, object]) -> Optional[Dict[str, object]]:
        teacher_alpha = self._current_alpha()
        teacher_action = self._teacher_to_action()
        if self.respawn_macro_active or self.respawn_pending or self._is_respawn_task(self.current_task):
            teacher_action = None

        chosen = None
        if (
            teacher_action
            and TEACHER_TARGET_PRIORITY
            and policy_action
            and policy_action.get("label") == "mouse_move"
            and (teacher_action.get("target_norm") or teacher_action.get("target_label"))
        ):
            chosen = teacher_action
        if self.stage0_enabled and teacher_action:
            chosen = teacher_action
        elif teacher_action and teacher_alpha > 0:
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
            # If we are heavily relying on the teacher (early training) but have no teacher action,
            # wait, but not forever. Random fallback helps break the cycle.
            if teacher_alpha > 0.8 and not teacher_action and self.rng.random() > 0.3:
                logger.debug("Waiting for teacher instruction (alpha=%.2f)", teacher_alpha)
                return {"label": "wait", "reason": "waiting_for_teacher"}
            return None

        label = str(chosen.get("label") or "").lower()
        macro_flag = chosen.get("macro") == "respawn"
        if not macro_flag:
            if self._should_suppress_shop():
                logger.info("SHOP suppress active -> forcing idle move")
                return {"label": "wait"}
            if label.startswith("click") and self._scene_is_forbidden():
                self.forbidden_until = max(self.forbidden_until, time.time() + max(0.0, POLICY_FORBIDDEN_COOLDOWN))
                logger.warning(
                    "Policy suppressed %s due to forbidden UI; substituting wait",
                    label,
                )
                return {"label": "wait"}

            if POLICY_COMBAT_AIM and self._enemies_present() and not self._scene_dialog_present():
                enemies = (self.latest_state or {}).get("enemies") or []
                target_center = self._pick_enemy_target(enemies)
                if target_center and not self._cursor_near_target(target_center[0], target_center[1]):
                    move = self._teacher_target_move(target_center)
                    if move:
                        logger.info("Combat aim -> moving cursor toward enemy")
                        return move

            if (
                POLICY_ENEMY_SKILL_SUBSTITUTE
                and self._enemies_present()
                and not self._scene_dialog_present()
                # Higher bias towards skills during combat
                and (label in {"click_primary", "click_secondary", "mouse_move", "wait"} or self.rng.random() < 0.5)
                and self._enemy_skill_ready()
            ):
                skill_key = self._choose_skill_key()
                if skill_key:
                    self.last_enemy_skill_ts = time.time()
                    self._record_skill_try(skill_key, self.latest_state or {})
                    logger.info("COMBAT: enemies present -> substituting '%s' with key_press '%s'", label, skill_key)
                    # Include target_norm so the bridge can aim if supported
                    payload = {"label": "key_press", "key": skill_key, "source": "enemy_skill_substitute"}
                    if chosen.get("target_norm"):
                        payload["target_norm"] = chosen["target_norm"]
                    return payload

            if label == "click_primary" and not self._click_ready():
                if self.active_target and self._cursor_is_fresh():
                    center = self.active_target.get("center") or self.active_target.get("target_norm")
                    move = self._teacher_target_move(center) if center else None
                    if move:
                        logger.info(
                            "Delaying click; moving cursor toward target %s",
                            center,
                        )
                        return move
                fallback = (
                    policy_action
                    if policy_action and policy_action.get("label") == "mouse_move"
                    else {"label": "wait"}
                )
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
        if self.respawn_macro_active:
            return None
        if not self.teacher_action:
            return None
        action_obj = self.teacher_action.get("action")
        if isinstance(action_obj, dict):
            structured = self._normalize_teacher_action_obj(action_obj)
            if structured:
                return structured
        raw_text = self.teacher_action.get("text") or ""
        if not raw_text:
            return None
        if self._text_hits_forbidden(raw_text):
            logger.info("Ignoring teacher action referencing forbidden target: %s", raw_text)
            return None
        text = raw_text.lower()

        if self._teacher_respawn_hint(text):
            logger.info("Starting respawn macro from teacher hint: %s", raw_text)
            self._start_respawn_macro()
            self.teacher_action = None
            return None

        target_norm = self.teacher_action.get("target_norm")
        target_label = self.teacher_action.get("target_label")
        target_move = self._teacher_target_move(target_norm) if target_norm else None

        if any(word in text for word in ("click", "attack", "shoot", "select")):
            button = "click_secondary" if "right" in text else "click_primary"
            if target_norm:
                x_norm, y_norm = target_norm
                if not self._cursor_is_fresh() or not self._cursor_near_target(x_norm, y_norm):
                    if target_move:
                        logger.info(
                            "Teacher requested %s but cursor not on target %s; moving cursor first",
                            button,
                            target_norm,
                        )
                        return target_move
                payload = {"label": button, "target_norm": target_norm}
                if target_label:
                    payload["target_label"] = target_label
                return payload
            return {"label": button}

        auto_click = self._maybe_teacher_autoclick(target_norm, target_label, text)
        if auto_click:
            return auto_click

        if "hold" in text and "mouse" in text:
            return {"label": "mouse_hold", "button": "left" if "left" in text else "right"}

        if "release" in text and "mouse" in text:
            return {"label": "mouse_release", "button": "left" if "left" in text else "right"}

        if self._teacher_wait_instruction(text):
            return {"label": "wait"}

        if target_move and any(term in text for term in ("move", "cursor", "hover", "aim", "place")):
            auto_click = self._maybe_teacher_autoclick(target_norm, target_label, text)
            if auto_click:
                return auto_click
            return target_move

        key = self._teacher_key_from_text(text)
        if key:
            if not self._teacher_key_allowed(key):
                logger.info("Teacher requested key '%s' not confirmed; ignoring", key)
                return None
            return {"label": "key_press", "key": key}

        move_vec = self._direction_from_text(text)
        if move_vec:
            dx, dy = move_vec
            return {"label": "mouse_move", "dx": dx, "dy": dy}

        key = self._extract_key(text)
        if key:
            if not self._teacher_key_allowed(key):
                logger.info("Teacher extracted key '%s' not confirmed; ignoring", key)
                return None
            return {"label": "key_press", "key": key}

        if target_move:
            auto_click = self._maybe_teacher_autoclick(target_norm, target_label, text)
            if auto_click:
                return auto_click
            return target_move

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

    def _key_allowed(self, key: Optional[str]) -> bool:
        if not key:
            return False
        if not self.profile_allowed_keys:
            return True
        return str(key).lower() in self.profile_allowed_keys

    def _teacher_key_confirmed(self, key: str) -> bool:
        if TEACHER_KEY_CONFIRM_HITS <= 0:
            return True
        stats = self.skill_stats.get(key)
        if not stats:
            return False
        try:
            hits = float(stats.get("hits", 0.0) or 0.0)
        except (TypeError, ValueError):
            hits = 0.0
        return hits >= TEACHER_KEY_CONFIRM_HITS

    def _teacher_key_allowed(self, key: Optional[str]) -> bool:
        if not key:
            return False
        key_norm = str(key).lower()
        if key_norm in self.profile_allowed_keys:
            return True
        return self._teacher_key_confirmed(key_norm)

    def _skill_key_confirmed(self, key: str) -> bool:
        if not key:
            return False
        key_norm = str(key).lower()
        if key_norm in self.profile_allowed_keys:
            return True
        if key_norm.isdigit():
            return True
        return self._teacher_key_confirmed(key_norm)

    def _enemies_present(self) -> bool:
        state = self.latest_state or {}
        # 1. Check explicit enemies list
        enemies = state.get("enemies") or []
        if isinstance(enemies, list) and enemies:
            for enemy in enemies:
                try:
                    conf = float((enemy or {}).get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    conf = 0.0
                if conf >= POLICY_ENEMY_MIN_CONF:
                    return True
        
        # 2. Check enemy health bars
        bars = state.get("enemy_bars") or []
        if isinstance(bars, list) and bars:
            return True

        # 3. Check objects list for anything labeled as enemy/monster/boss
        objects = state.get("objects") or []
        for obj in objects:
            label = str(obj.get("label") or obj.get("class") or "").lower()
            if any(kw in label for kw in ("enemy", "monster", "boss", "hostile", "bandit")):
                try:
                    conf = float(obj.get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    conf = 0.0
                if conf >= POLICY_ENEMY_MIN_CONF:
                    return True

        # 4. Check global stats
        stats = state.get("stats") or {}
        try:
            if float(stats.get("enemy_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass

        # 5. Check OCR text for enemy-related keywords (with quest filtering)
        texts = self._collect_scene_texts(state)
        quest_indicators = {"at the gate", "objective", "quest", "completed", "mission", "menu", "gate"}
        quest_compact = {"objective", "quest", "completed", "mission", "menu", "gate"}
        enemy_keywords = ("enemy", "monster", "boss", "hostile", "bandit")
        for entry in texts:
            lowered = str(entry).lower()
            if "prompt_" in lowered:
                continue
            compact = "".join(ch for ch in lowered if ch.isalnum())
            # Ignore quest/UI headers even if OCR merges spaces.
            if any(q in lowered for q in quest_indicators) or any(q in compact for q in quest_compact):
                continue
            # Only short labels should trigger combat.
            if len(compact) > 24 or len(lowered.split()) > 3:
                continue
            if any(kw in lowered or kw in compact for kw in enemy_keywords):
                return True

        return False

    def _enemy_skill_ready(self) -> bool:
        if POLICY_ENEMY_SKILL_MIN_INTERVAL <= 0:
            return True
        return (time.time() - self.last_enemy_skill_ts) >= POLICY_ENEMY_SKILL_MIN_INTERVAL

    def _scene_dialog_present(self) -> bool:
        state = self.latest_state or {}
        flags = state.get("flags") or {}
        if flags.get("death"):
            return True
        if self._latest_scene_has_respawn_text():
            return True
        scores = state.get("prompt_scores") or {}
        try:
            score = float(scores.get("dialog", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        return score >= POLICY_DIALOG_SCORE_MIN

    @staticmethod
    def _intent_priority(intent: Optional[str]) -> int:
        priorities = {
            "idle": 0,
            "recover": 1,
            "combat": 2,
            "interact": 3,
            "navigate": 4,
            "observe": 5,
        }
        return priorities.get(str(intent or "").lower(), 99)

    @staticmethod
    def _labels_present(items: Optional[list], labels: Set[str]) -> bool:
        if not items or not labels:
            return False
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_label = (
                item.get("label")
                or item.get("class")
                or item.get("name")
                or item.get("title")
                or ""
            )
            normalized = _normalize_phrase(str(raw_label))
            if normalized and normalized in labels:
                return True
        return False

    @staticmethod
    def _prompt_max_score(scores: Optional[dict], labels: Set[str]) -> float:
        if not isinstance(scores, dict) or not labels:
            return 0.0
        best = 0.0
        for key, value in scores.items():
            normalized = _normalize_phrase(str(key))
            if normalized not in labels:
                continue
            try:
                score = float(value or 0)
            except (TypeError, ValueError):
                continue
            if score > best:
                best = score
        return best

    @staticmethod
    def _enemies_present_in_state(state: Optional[dict]) -> bool:
        if not isinstance(state, dict):
            return False
        enemies = state.get("enemies") or []
        if isinstance(enemies, list) and enemies:
            if POLICY_ENEMY_MIN_CONF <= 0:
                return True
            for enemy in enemies:
                try:
                    conf = float((enemy or {}).get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    conf = 0.0
                if conf >= POLICY_ENEMY_MIN_CONF:
                    return True
        stats = state.get("stats") or {}
        try:
            return float(stats.get("enemy_count", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    def _infer_intent(self, state: Optional[dict]) -> Tuple[str, str]:
        if not isinstance(state, dict):
            return "observe", "no_state"
        flags = state.get("flags") or {}
        if flags.get("in_game") is False:
            return "idle", "not_in_game"
        if flags.get("death"):
            return "recover", "death_flag"
        texts = self._collect_scene_texts(state)
        if texts and self._scene_has_respawn_text(texts):
            return "recover", "respawn_text"
        scores = state.get("prompt_scores") or {}
        if self._prompt_max_score(scores, {"dialog"}) >= self.intent_dialog_score:
            return "recover", "dialog_prompt"
        if self._enemies_present_in_state(state):
            return "combat", "enemy_list"
        if (
            self._labels_present(state.get("targets") or [], self.intent_combat_labels)
            or self._labels_present(state.get("objects") or [], self.intent_combat_labels)
        ):
            return "combat", "enemy_label"
        # Prompt-only combat is noisy in many games (quest headers, UI text).
        # Require an additional enemy signal from scene heuristics/OCR filters.
        if (
            self._prompt_max_score(scores, self.intent_combat_labels) >= self.intent_enemy_score
            and self._enemies_present()
        ):
            return "combat", "enemy_prompt"
        if (
            self._labels_present(state.get("targets") or [], self.intent_interact_labels)
            or self._labels_present(state.get("objects") or [], self.intent_interact_labels)
            or self._prompt_max_score(scores, self.intent_interact_labels) >= self.intent_interact_score
        ):
            return "interact", "interact_signal"
        if (
            self._labels_present(state.get("targets") or [], self.intent_nav_labels)
            or self._labels_present(state.get("objects") or [], self.intent_nav_labels)
            or self._prompt_max_score(scores, self.intent_nav_labels) >= self.intent_nav_score
        ):
            return "navigate", "navigate_signal"
        return "observe", "default"

    def _update_intent(self, state: Optional[dict]) -> str:
        if not self.intent_enabled:
            return self.current_intent or "observe"
        desired, reason = self._infer_intent(state)
        now = time.time()
        if self.current_intent is None:
            self.current_intent = desired
            self.intent_reason = reason
            self.intent_last_change_ts = now
            return desired
        if desired == self.current_intent:
            self.intent_reason = reason
            return desired
        if self.current_intent == "idle" and desired != "idle":
            self.current_intent = desired
            self.intent_reason = reason
            self.intent_last_change_ts = now
            return desired
        elapsed = now - self.intent_last_change_ts
        if self.intent_min_sec <= 0 or elapsed >= self.intent_min_sec:
            self.current_intent = desired
            self.intent_reason = reason
            self.intent_last_change_ts = now
            return desired
        if self._intent_priority(desired) < self._intent_priority(self.current_intent):
            self.current_intent = desired
            self.intent_reason = reason
            self.intent_last_change_ts = now
            return desired
        return self.current_intent

    def _teacher_key_from_text(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:press|tap|hit|use)\s+(?:key\s+)?(shift)",
            r"(?:press|tap|hit|use)\s+(?:key\s+)?(ctrl)",
            r"(?:press|tap|hit|use)\s+(?:key\s+)?(alt)",
            r"(?:press|tap|hit|use)\s+(?:key\s+)?(space|enter|escape|tab)",
            r"(?:press|tap|hit|use)\s+(?:key\s+)?([wasdqerf])",
            r"(?:use|drink)\s+(?:flask|potion)\s*(\d)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                key = match.group(1)
                if len(key) == 1 and key.isalpha():
                    return key.lower()
                return key.lower()
        return None

    def _teacher_action_repr(self, action: dict) -> str:
        label = str(action.get("label") or action.get("action") or "").lower()
        if label == "key_press":
            key = action.get("key") or ""
            return f"press {key}".strip()
        if label == "click_secondary":
            return "click right"
        if label == "click_primary":
            return "click left"
        if label == "mouse_move":
            return "move mouse"
        return label or "wait"

    def _normalize_teacher_label(self, label: str) -> str:
        label = str(label or "").strip().lower()
        aliases = {
            "click": "click_primary",
            "attack": "click_primary",
            "primary": "click_primary",
            "secondary": "click_secondary",
            "right_click": "click_secondary",
            "left_click": "click_primary",
            "move_mouse": "mouse_move",
        }
        return aliases.get(label, label)

    def _normalize_teacher_action_obj(self, action: dict) -> Optional[Dict[str, object]]:
        label = self._normalize_teacher_label(action.get("label") or action.get("action") or action.get("type"))
        if not label:
            return None
        target_norm = action.get("target_norm") or action.get("target")
        if isinstance(target_norm, dict):
            try:
                target_norm = [float(target_norm.get("x")), float(target_norm.get("y"))]
            except (TypeError, ValueError):
                target_norm = None
        if isinstance(target_norm, (list, tuple)) and len(target_norm) >= 2:
            try:
                x_norm = max(0.0, min(1.0, float(target_norm[0])))
                y_norm = max(0.0, min(1.0, float(target_norm[1])))
                target_norm = [x_norm, y_norm]
            except (TypeError, ValueError):
                target_norm = None
        else:
            target_norm = None
        target_label = action.get("target_label")
        if label == "key_press":
            key = str(action.get("key") or "").lower().strip()
            if not key:
                return None
            if not self._teacher_key_allowed(key):
                logger.info("Teacher requested key '%s' not confirmed; ignoring", key)
                return None
            return {"label": "key_press", "key": key}
        if label in {"click_primary", "click_secondary"}:
            if target_norm:
                if not self._cursor_is_fresh() or not self._cursor_near_target(*target_norm):
                    move = self._teacher_target_move(target_norm)
                    if move:
                        logger.info(
                            "Teacher requested %s but cursor not on target %s; moving cursor first",
                            label,
                            target_norm,
                        )
                        return move
            payload = {"label": label}
            if target_norm:
                payload["target_norm"] = target_norm
            if target_label:
                payload["target_label"] = target_label
            return payload
        if label == "mouse_move":
            dx = action.get("dx")
            dy = action.get("dy")
            if isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
                return {"label": "mouse_move", "dx": int(dx), "dy": int(dy)}
            if target_norm:
                move = self._teacher_target_move(target_norm)
                if move:
                    return move
            return {"label": "wait"}
        if label in {"mouse_hold", "mouse_release"}:
            button = action.get("button") or "left"
            return {"label": label, "button": button}
        if label == "wait":
            return {"label": "wait"}
        return None

    @staticmethod
    def _teacher_respawn_hint(text: str) -> bool:
        return "resurrect at checkpoint" in text

    def _teacher_wait_instruction(self, text: str) -> bool:
        return any(phrase in text for phrase in ("wait", "observe", "hold still", "pause"))

    def _action_from_task(self, state: dict) -> Dict[str, object]:
        """Turn GOAP tasks into concrete mouse/keyboard commands.

        MOVE_TO / CLICK_BUTTON tasks use normalized coordinates published by
        GOAP (e.g. dialog buttons). We project those normalized values onto the
        configured capture resolution (POLICY_SCREEN_WIDTH/HEIGHT) and apply
        optional offsets (POLICY_OFFSET_X/Y) to align with the remote desktop.
        """
        task = self.current_task or {}
        action_type = (task.get("action_type") or "MOVE_TO").upper()
        target = task.get("target") or {}
        dx = target.get("x", 0.5) - 0.5
        dy = target.get("y", 0.5) - 0.5
        scale = MOUSE_RANGE
        x_norm = target.get("x_norm")
        y_norm = target.get("y_norm")
        if x_norm is None and target.get("x") is not None:
            raw_x = float(target.get("x"))
            x_norm = raw_x if 0.0 <= raw_x <= 1.0 else min(1.0, max(0.0, raw_x))
        if y_norm is None and target.get("y") is not None:
            raw_y = float(target.get("y"))
            y_norm = raw_y if 0.0 <= raw_y <= 1.0 else min(1.0, max(0.0, raw_y))

        def _move_payload():
            if x_norm is None or y_norm is None:
                return {
                    "label": "mouse_move",
                    "dx": int(dx * scale),
                    "dy": int(dy * scale),
                    "task_id": task.get("task_id"),
                }
            target_px, cursor_px, (delta_x, delta_y) = compute_cursor_motion(
                x_norm,
                y_norm,
                self.cursor_x_norm,
                self.cursor_y_norm,
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                CURSOR_OFFSET_X,
                CURSOR_OFFSET_Y,
            )
            if self._is_forbidden_target(target):
                logger.info(
                    "Policy skipping move to forbidden target tags=%s", target.get("tags")
                )
                return {
                    "label": "mouse_move",
                    "dx": 0,
                    "dy": 0,
                    "task_id": task.get("task_id"),
                }
            payload = {
                "label": "mouse_move",
                "dx": delta_x,
                "dy": delta_y,
                "task_id": task.get("task_id"),
                "target_norm": [x_norm, y_norm],
                "target_px": target_px,
            }
            if self.stage0_enabled or target.get("area") == "critical_dialog":
                scope = target.get("scope") or "generic_ui"
                profile = target.get("profile") or "unknown"
                source = target.get("target_source") or "policy"
                logger.info(
                    "Dialog MOVE task=%s stage=%s scope=%s profile=%s source=%s target_norm=(%.3f,%.3f) target_px=%s cursor_px=%s delta=(%s,%s)",
                    task.get("task_id"),
                    LEARNING_STAGE,
                    scope,
                    profile,
                    source,
                    x_norm,
                    y_norm,
                    target_px,
                    cursor_px,
                    delta_x,
                    delta_y,
                )
            return payload

        if action_type == "WAIT":
            duration = float(task.get("duration", 0.0) or 0.0)
            started_at = task.get("started_at")
            now = time.time()
            if started_at is None:
                task["started_at"] = now
                started_at = now
            elapsed = now - float(started_at)
            complete = duration <= 0.0 or elapsed >= duration
            if complete:
                task["completed"] = True
            return {"label": "wait", "task_id": task.get("task_id"), "task_complete": complete}
        if action_type == "MOVE_TO":
            if POLICY_CLICK_TO_MOVE and x_norm is not None and y_norm is not None:
                return {
                    "label": "click_primary",
                    "task_id": task.get("task_id"),
                    "confidence": 1.0,
                    "target_norm": [x_norm, y_norm],
                }
            return _move_payload()
        if action_type == "ATTACK_TARGET":
            payload = {"label": "click_primary", "task_id": task.get("task_id"), "confidence": 1.0}
            if x_norm is not None and y_norm is not None:
                payload["target_norm"] = [x_norm, y_norm]
            return payload
        if action_type == "LOOT_NEARBY":
            return {"label": "click_secondary", "task_id": task.get("task_id"), "confidence": 1.0}
        if action_type == "CLICK_BUTTON":
            if self.stage0_enabled and x_norm is not None and y_norm is not None and task.get("task_id") not in self.stage0_move_targets:
                self.stage0_move_targets[task.get("task_id")] = True
                return _move_payload()
            button = str(target.get("button", "primary")).lower()
            label = "click_secondary" if "right" in button else "click_primary"
            if self._is_forbidden_target(target):
                logger.info("Policy blocked click on forbidden target tags=%s", target.get("tags"))
                return {
                    "label": "mouse_move",
                    "dx": 0,
                    "dy": 0,
                    "task_id": task.get("task_id"),
                }
            if self._hover_hits_forbidden_text():
                logger.info("Hover indicates forbidden UI (shop); reissuing move task=%s", task.get("task_id"))
                return _move_payload()
            if x_norm is not None and y_norm is not None:
                if not self._cursor_is_fresh() or not self._cursor_near_target(x_norm, y_norm):
                    logger.info(
                        "Cursor not settled on target task=%s cursor=(%.3f,%.3f) target=(%.3f,%.3f)",
                        task.get("task_id"),
                        self.cursor_x_norm,
                        self.cursor_y_norm,
                        x_norm,
                        y_norm,
                    )
                    return _move_payload()
                if not self._hover_confirms_target(task, target):
                    logger.info(
                        "Hover text mismatch, delaying click task=%s target_text=%s",
                        task.get("task_id"),
                        target.get("text") or target.get("text_hint"),
                    )
                    return _move_payload()
            if target.get("area") == "critical_dialog":
                scope = target.get("scope") or "generic_ui"
                profile = target.get("profile") or "unknown"
                source = target.get("target_source") or "policy"
                logger.info(
                    "Dialog CLICK task=%s stage=%s scope=%s profile=%s source=%s type=%s target_norm=%s",
                    task.get("task_id"),
                    LEARNING_STAGE,
                    scope,
                    profile,
                    source,
                    label,
                    target.get("target_norm") or [x_norm, y_norm],
                )
            payload = {"label": label, "task_id": task.get("task_id"), "confidence": 1.0}
            if x_norm is not None and y_norm is not None:
                payload["target_norm"] = [x_norm, y_norm]
            return payload
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
            if not self._key_allowed(key):
                logger.info("Policy suppressed key '%s' (not in allowed_keys)", key)
                return {"label": "wait"}
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

    def _start_respawn_macro(self) -> None:
        now = time.time()
        self.respawn_macro_queue.clear()
        delay = 0.0
        move_steps = max(1, RESPAWN_MACRO_MOVE_STEPS)
        for _ in range(move_steps):
            self.respawn_macro_queue.append((now + delay, {"macro_template": "move"}))
            delay += max(0.01, RESPAWN_MACRO_MOVE_DELAY)
        delay += max(0.0, RESPAWN_MACRO_PRE_CLICK_DELAY)
        click_delay = max(0.05, RESPAWN_MACRO_CLICK_DELAY)
        for _ in range(max(1, RESPAWN_MACRO_CLICK_COUNT)):
            self.respawn_macro_queue.append(
                (
                    now + delay,
                    {
                        "label": "click_primary",
                        "macro": "respawn",
                    },
                )
            )
            delay += click_delay
        self.respawn_macro_block_until = now + delay + max(0.5, RESPAWN_MACRO_SETTLE_SEC)
        self.respawn_macro_active = True
        logger.info(
            "Respawn macro queued: %s move steps, %s clicks, pre_click=%.2fs settle=%.2fs",
            move_steps,
            max(1, RESPAWN_MACRO_CLICK_COUNT),
            RESPAWN_MACRO_PRE_CLICK_DELAY,
            RESPAWN_MACRO_SETTLE_SEC,
        )
        self.stage0_pending = False
        logger.info(
            "Respawn macro queued: %s move steps, %s clicks, settle %.2fs",
            move_steps,
            max(1, RESPAWN_MACRO_CLICK_COUNT),
            RESPAWN_MACRO_SETTLE_SEC,
        )

    def _next_respawn_macro_action(self) -> Optional[Dict[str, object]]:
        if not self.respawn_macro_active:
            return None
        now = time.time()
        if self.respawn_macro_queue:
            when, template = self.respawn_macro_queue[0]
            if now < when:
                return None
            self.respawn_macro_queue.popleft()
            return self._resolve_respawn_action(template)
        if now >= self.respawn_macro_block_until:
            self.respawn_macro_active = False
            logger.info("Respawn macro finished")
        return None

    def _resolve_respawn_action(self, template: Dict[str, object]) -> Dict[str, object]:
        if template.get("macro_template") == "move":
            return self._respawn_move_payload()
        action = dict(template)
        action.setdefault("macro", "respawn")
        return action

    def _respawn_move_payload(self) -> Dict[str, object]:
        target_px, cursor_px, (delta_x, delta_y) = compute_cursor_motion(
            RESPAWN_TARGET_X,
            RESPAWN_TARGET_Y,
            self.cursor_x_norm,
            self.cursor_y_norm,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            CURSOR_OFFSET_X,
            CURSOR_OFFSET_Y,
        )
        action = {
            "label": "mouse_move",
            "dx": delta_x,
            "dy": delta_y,
            "target_norm": [RESPAWN_TARGET_X, RESPAWN_TARGET_Y],
            "target_px": target_px,
            "cursor_px": cursor_px,
            "macro": "respawn",
        }
        return action

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
        label = str(chosen.get("label") or "").lower()
        if label == "mouse_move":
            self.consecutive_moves += 1
        elif label in {"click_primary", "click_secondary", "click_middle", "key_press"}:
            self.consecutive_moves = 0
            
        key_payload = self._key_payload(chosen)
        act_payload = {"action": label}
        if label == "mouse_move":
            act_payload["dx"] = int(chosen.get("dx", 0))
            act_payload["dy"] = int(chosen.get("dy", 0))
            if "target_norm" in chosen:
                act_payload["target_norm"] = chosen.get("target_norm")
            if "target_px" in chosen:
                act_payload["target_px"] = chosen.get("target_px")
        elif label in {"click_primary", "click_secondary", "click_middle"}:
            if "target_norm" in chosen:
                act_payload["target_norm"] = chosen.get("target_norm")
            if "target_px" in chosen:
                act_payload["target_px"] = chosen.get("target_px")
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
            "intent": self.current_intent,
            "intent_reason": self.intent_reason,
            "timestamp": time.time(),
        }
        if self.current_task:
            envelope["task_id"] = self.current_task.get("task_id")
            envelope["goal_id"] = self.current_task.get("goal_id")
        full_payload = {**envelope, **act_payload}
        if envelope.get("task_id"):
            logger.info(
                "Policy action=%s task=%s goal=%s stage=%s",
                label,
                envelope.get("task_id"),
                envelope.get("goal_id"),
                LEARNING_STAGE,
            )
        logger.info("Stage0 publish action topic=%s payload=%s", ACT_CMD_TOPIC, full_payload)
        client.publish(ACT_CMD_TOPIC, json.dumps(full_payload))
        if key_payload:
            client.publish(CONTROL_TOPIC, json.dumps(key_payload))
        logger.info("Published blended action: %s", label)
        if label and str(label).startswith("click"):
            self.last_click_ts = time.time()
        self.last_label = label
        if label:
            self.recent_action_labels.append(str(label))
        if self.stage0_enabled and label and label != "wait":
            self.stage0_pending = True
            self.stage0_reference = self.stage0_last_signature
            self.stage0_wait_until = time.time() + STAGE0_SETTLE_SEC
        self._update_cursor_estimate(chosen)
        self._maybe_advance_task(chosen)
        self.feedback_pending = True
        self.feedback_signature = self._state_signature(self.latest_state or {})

    def _control_action_signature(self, action: Dict[str, object]) -> Dict[str, object]:
        label = action.get("label") or action.get("action") or ""
        vector = None
        if isinstance(action.get("dx"), (int, float)) or isinstance(action.get("dy"), (int, float)):
            vector = (float(action.get("dx", 0.0)), float(action.get("dy", 0.0)))
        return {"label": str(label), "vector": vector}

    def _record_control_metrics(self, client, tick_ms: float, action: Optional[Dict[str, object]]):
        ready = action is not None
        if SLA_TICK_MS:
            ready = ready and tick_ms <= SLA_TICK_MS
        self.control_ready_window.append(1 if ready else 0)
        self.control_tick_count += 1
        if self.control_tick_count % CONTROL_METRIC_SAMPLE_EVERY == 0:
            tick_ok = tick_ms <= SLA_TICK_MS if SLA_TICK_MS else None
            emit_control_metric(
                client,
                "control/tick_ms",
                tick_ms,
                ok=tick_ok,
                tags={"agent": "policy_agent", "sample_every": CONTROL_METRIC_SAMPLE_EVERY},
            )
            if self.control_ready_window:
                ratio = sum(self.control_ready_window) / len(self.control_ready_window)
                emit_control_metric(
                    client,
                    "control/next_chunk_ready_ratio",
                    ratio,
                    tags={"window": len(self.control_ready_window), "sample_every": CONTROL_METRIC_SAMPLE_EVERY},
                )
        if not action:
            return
        current = self._control_action_signature(action)
        prev = self.last_control_action
        self.last_control_action = current
        if not prev or prev.get("label") == current.get("label"):
            return
        prev_vec = prev.get("vector")
        curr_vec = current.get("vector")
        if prev_vec is None or curr_vec is None:
            return
        jerk = math.hypot(curr_vec[0] - prev_vec[0], curr_vec[1] - prev_vec[1])
        jerk_ok = jerk <= SLA_JERK_MAX if SLA_JERK_MAX else None
        emit_control_metric(
            client,
            "control/chunk_boundary_jerk",
            jerk,
            ok=jerk_ok,
            tags={"from": prev.get("label"), "to": current.get("label")},
        )

    # ----------------------------------------------------------------- Public
    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 60)
        self.client.loop_forever()

    def _state_signature(self, state: dict) -> str:
        texts = "|".join(self._collect_scene_texts(state))
        objects = state.get("objects") or []
        flags = state.get("flags") or {}
        death = "1" if flags.get("death") else "0"
        return f"{texts}|{len(objects)}|{death}"

    def _update_cursor_estimate(self, action: Dict[str, object]):
        label = (action.get("label") or "").lower()
        if label != "mouse_move":
            return
        target_norm = action.get("target_norm")
        has_target = isinstance(target_norm, list) and len(target_norm) == 2
        if self._cursor_is_fresh() and not has_target:
            return
        if isinstance(target_norm, list) and len(target_norm) == 2:
            self.cursor_x_norm = float(target_norm[0])
            self.cursor_y_norm = float(target_norm[1])
        else:
            dx = int(action.get("dx", 0))
            dy = int(action.get("dy", 0))
            self.cursor_x_norm = min(1.0, max(0.0, self.cursor_x_norm + dx / max(1, SCREEN_WIDTH)))
            self.cursor_y_norm = min(1.0, max(0.0, self.cursor_y_norm + dy / max(1, SCREEN_HEIGHT)))

    def _maybe_advance_task(self, action: Dict[str, object]):
        if not self.current_task:
            return
        action_type = (self.current_task.get("action_type") or "").upper()
        label = (action.get("label") or "").lower()
        task_id = self.current_task.get("task_id")
        completed = False
        if action_type == "MOVE_TO":
            if label == "mouse_move":
                completed = True
            elif POLICY_CLICK_TO_MOVE and label.startswith("click"):
                completed = True
        elif action_type == "WAIT":
            if action.get("task_complete") or self.current_task.get("completed"):
                completed = True
        elif action_type in {"ATTACK_TARGET", "LOOT_NEARBY", "CLICK_BUTTON"} and label.startswith("click"):
            completed = True
        if completed:
            logger.info("Completed task %s (%s)", task_id, action_type)
            if task_id in self.stage0_move_targets:
                del self.stage0_move_targets[task_id]
            if self.task_queue and self.task_queue[0].get("task_id") == task_id:
                self.task_queue.popleft()
            self.current_task = self.task_queue[0] if self.task_queue else None
            if not self.current_task and self.respawn_pending:
                self.respawn_pending = False


def main():
    agent = PolicyAgent()
    agent.start()


if __name__ == "__main__":
    main()
