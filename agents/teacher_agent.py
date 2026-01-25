#!/usr/bin/env python3
"""Teacher agent that queries an LLM for high-level guidance."""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set

try:
    import requests
except ImportError:
    requests = None

import paho.mqtt.client as mqtt

try:
    from .mem_rpc import MemRPC
except ImportError:
    from mem_rpc import MemRPC
from control_profile import load_profile, safe_profile
try:
    from .llm_gate import acquire_gate, blocked_reason, release_gate
except ImportError:
    from llm_gate import acquire_gate, blocked_reason, release_gate

# --- Setup ---
logging.basicConfig(level=os.getenv("TEACHER_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("teacher_agent")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST, MQTT_PORT = os.getenv("MQTT_HOST", "127.0.0.1"), int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC, SIM_TOPIC = os.getenv("SCENE_TOPIC", "scene/state"), os.getenv("SIM_TOPIC", "sim_core/state")
SNAPSHOT_TOPIC, ACT_RESULT_TOPIC = os.getenv("SNAPSHOT_TOPIC", "vision/snapshot"), os.getenv("ACT_RESULT_TOPIC", "act/result")
GAME_IDENTITY_TOPIC = os.getenv("GAME_IDENTITY_TOPIC", "game/identity")
TEACHER_TOPIC, MEM_STORE_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action"), os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_QUERY_TOPIC, MEM_REPLY_TOPIC, MEM_RESPONSE_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query"), os.getenv("MEM_REPLY_TOPIC", "mem/reply"), os.getenv("MEM_RESPONSE_TOPIC", "mem/response")
GOAL_TOPIC, GUIDE_KEY = os.getenv("GOAL_TOPIC", "goals/high_level"), os.getenv("TEACHER_GUIDE_KEY", "teacher_guides")
TEACHER_PROVIDER, OPENAI_MODEL = os.getenv("TEACHER_PROVIDER", "openai").lower(), os.getenv("TEACHER_OPENAI_MODEL", "gpt-4o-mini")
LOCAL_ENDPOINT, LOCAL_TIMEOUT = os.getenv("TEACHER_LOCAL_ENDPOINT"), float(os.getenv("TEACHER_LOCAL_TIMEOUT", "20"))
MAX_HISTORY, REQUEST_INTERVAL = int(os.getenv("TEACHER_HISTORY", "6")), float(os.getenv("TEACHER_INTERVAL_SEC", "2.0"))
MAX_ATTEMPTS, TEMPERATURE = int(os.getenv("TEACHER_MAX_ATTEMPTS", "3")), float(os.getenv("TEACHER_TEMPERATURE", "0.2"))
LLM_GATE_LOG_SEC = float(os.getenv("LLM_GATE_LOG_SEC", "15"))
LEARNING_STAGE = int(os.getenv("LEARNING_STAGE", "1"))
DEATH_ACTION_TEXT = os.getenv("TEACHER_DEATH_ACTION", "Click 'Resurrect at Checkpoint'")
DEATH_GOAL_HINT, DEATH_GOAL_TYPE = os.getenv("TEACHER_DEATH_HINT", "Death dialog detected; respawn immediately."), os.getenv("TEACHER_DEATH_GOAL_TYPE", "respawn")
STAGE0_SUMMARY_HINT = os.getenv("TEACHER_STAGE0_SUMMARY", "Stage-0 exploration: describe UI precisely and identify a safe experiment.")
STAGE0_ACTION_HINT = os.getenv("TEACHER_STAGE0_ACTION", "Suggest one safe, deliberate action and remind agent to wait for feedback.")
TEACHER_RULE_LIMIT, TEACHER_RECENT_CRITICAL_LIMIT = int(os.getenv("TEACHER_RULE_LIMIT", "5")), int(os.getenv("TEACHER_RECENT_CRITICAL_LIMIT", "3"))
TEACHER_REPETITION_LIMIT, TEACHER_BACKOFF_SEC = int(os.getenv("TEACHER_REPETITION_LIMIT", "3")), float(os.getenv("TEACHER_BACKOFF_SEC", "30"))
TEACHER_MAX_FAIL_STREAK = int(os.getenv("TEACHER_MAX_FAIL_STREAK", "5"))
TEACHER_STAGE0_MAX_TOKENS, TEACHER_STAGE0_TEXT_LIMIT = int(os.getenv("TEACHER_STAGE0_MAX_TOKENS", "128")), int(os.getenv("TEACHER_STAGE0_TEXT_LIMIT", "256"))
TEACHER_CONTEXT_KEY, TEACHER_CONTEXT_REFRESH_SEC = os.getenv("TEACHER_CONTEXT_KEY", "teacher_context"), float(os.getenv("TEACHER_CONTEXT_REFRESH_SEC", "7200"))
TEACHER_CONTEXT_MIN_CHARS, TEACHER_CONTEXT_TEXT_LIMIT = int(os.getenv("TEACHER_CONTEXT_MIN_CHARS", "24")), int(os.getenv("TEACHER_CONTEXT_TEXT_LIMIT", "768"))
TEACHER_CONTEXT_OBJECT_LIMIT = int(os.getenv("TEACHER_CONTEXT_OBJECT_LIMIT", "8"))
TEACHER_CONTEXT_SCOPE_FALLBACK = os.getenv("TEACHER_CONTEXT_SCOPE_FALLBACK", "generic_ui")
TEACHER_CONTEXT_DIFF_JACCARD, TEACHER_CONTEXT_DIFF_MIN_CHARS = float(os.getenv("TEACHER_CONTEXT_DIFF_JACCARD", "0.4")), int(os.getenv("TEACHER_CONTEXT_DIFF_MIN_CHARS", "48"))
TEACHER_PROMPT_TOP_K = int(os.getenv("TEACHER_PROMPT_TOP_K", "5"))
TEACHER_PROMPT_MIN_SCORE = float(os.getenv("TEACHER_PROMPT_MIN_SCORE", "0.02"))
TEACHER_PROMPT_ALLOW = {item.strip().lower() for item in os.getenv("TEACHER_PROMPT_ALLOW", "").split(",") if item.strip()}
TEACHER_REQUIRE_IN_GAME = os.getenv("TEACHER_REQUIRE_IN_GAME", "0") != "0"
TEACHER_REQUIRE_IN_GAME_STRICT = os.getenv("TEACHER_REQUIRE_IN_GAME_STRICT", "0") != "0"
TEACHER_ALLOW_OCR_GAME_ID = os.getenv("TEACHER_ALLOW_OCR_GAME_ID", "0") != "0"
TEACHER_GAME_KEYWORDS = {item.strip().lower() for item in os.getenv("TEACHER_GAME_KEYWORDS", "path of exile,poe,life,mana,inventory,quest,map").split(",") if item.strip()}
TEACHER_RESPAWN_KEYWORDS = {item.strip().lower() for item in os.getenv("TEACHER_RESPAWN_KEYWORDS", "resurrect,resurrect at checkpoint,respawn,revive").split(",") if item.strip()}
TEACHER_DEATH_SCOPES = {item.strip().lower() for item in os.getenv("TEACHER_DEATH_SCOPES", "death_dialog,critical_dialog:death").split(",") if item.strip()}
TEACHER_WATCHDOG_KEY, TEACHER_WATCHDOG_COOLDOWN = os.getenv("TEACHER_WATCHDOG_KEY", "teacher_watchdog"), float(os.getenv("TEACHER_WATCHDOG_COOLDOWN_SEC", "90"))
TEACHER_WATCH_RESPAWN_TERMS = {item.strip().lower() for item in os.getenv("TEACHER_WATCH_RESPAWN_TERMS", "resurrect at checkpoint,resurrect,respawn").split(",") if item.strip()}
TEACHER_WATCH_FORBID_TERMS = {item.strip().lower() for item in os.getenv("TEACHER_WATCH_FORBID_TERMS", "microtransaction shop,shop button").split(",") if item.strip()}
TEACHER_RESPAWN_RULE_ID, TEACHER_SHOP_RULE_ID = os.getenv("TEACHER_RESPAWN_RULE_ID", "rule_respawn_click"), os.getenv("TEACHER_SHOP_RULE_ID", "rule_avoid_shop")
TARGET_HINT_INSTRUCTIONS = ("Always identify the on-screen UI element the next action should interact with. "
                          "Estimate normalized screen coordinates in the [0.0, 1.0] range (0,0 is top-left). "
                          "End your reasoning with a line formatted exactly as 'TARGET_HINT: <short label> | target=(x.xx,y.yy)'. "
                          "If no pointer target is relevant, use label 'none' and target=(0.50,0.50).")
TEACHER_INCLUDE_CONTROLS = os.getenv("TEACHER_INCLUDE_CONTROLS", "1") != "0"
CONTROL_PROFILE_GAME_ID = os.getenv("TEACHER_CONTROL_PROFILE_GAME_ID") or os.getenv("CONTROL_PROFILE_GAME_ID") or os.getenv("RECORDER_GAME_ID") or os.getenv("GAME_ID") or "unknown_game"

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

def _text_fingerprint(value: str) -> str: return " ".join(str(value).strip().lower().split())
def _text_change_ratio(prev: str, current: str) -> float:
    if not prev and not current: return 0.0
    prev_tokens, curr_tokens = {token for token in (prev or "").split() if token}, {token for token in (current or "").split() if token}
    if not prev_tokens and not curr_tokens: return 0.0
    union = prev_tokens | curr_tokens
    if not union: return 0.0
    return 1.0 - (len(prev_tokens & curr_tokens) / len(union))

def _scene_in_game(scene: dict) -> bool:
    flags = scene.get("flags") or {}
    in_game = flags.get("in_game")
    if TEACHER_REQUIRE_IN_GAME_STRICT:
        return in_game is True
    return in_game is not False

def _normalize_game_id(value: str) -> str:
    if not value:
        return "unknown_game"
    lowered = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_game"

class BaseChatClient:
    def _complete(self, messages):  # pragma: no cover - subclasses implement transport
        raise NotImplementedError

    def summarize(self, scene_summary: str, snapshot_hint: str, recent_actions: str) -> str:
        prompt = [
            {"role": "system", "content": "Summarize the current UI state and constraints for the agent."},
            {
                "role": "user",
                "content": (
                    f"Scene summary:\n{scene_summary}\n\n"
                    f"Snapshot hint: {snapshot_hint}\n\n"
                    f"Recent actions:\n{recent_actions}\n\n"
                    "Return a concise summary the agent can act on."
                ),
            },
        ]
        return self._complete(prompt)

    def propose_action(self, reasoning: str, scene_summary: str, recent_actions: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Propose one safe, concrete next action. "
                    "Use only controls listed in the scene summary under Controls. "
                    "If you include coordinates, use normalized 0..1."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reasoning:\n{reasoning}\n\n"
                    f"Scene summary:\n{scene_summary}\n\n"
                    f"Recent actions:\n{recent_actions}\n\n"
                    "Return the action as a single sentence."
                ),
            },
        ]
        return self._complete(prompt)

    def describe_environment(self, scene_text: str, objects: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "Identify the game and key UI context as a compact JSON object.",
            },
            {
                "role": "user",
                "content": (
                    f"Scene text:\n{scene_text}\n\n"
                    f"Objects:\n{objects}\n\n"
                    "Return JSON with keys: game, summary, controls."
                ),
            },
        ]
        return self._complete(prompt)

class OpenAIChatClient(BaseChatClient):
    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        if not api_key: raise ValueError("OPENAI_API_KEY is required for teacher agent")
        try: import openai
        except ImportError as exc: raise RuntimeError("openai package is not installed") from exc
        self._openai, self._model = openai, model
        self._openai.api_key = api_key
    def _complete(self, messages):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._openai.ChatCompletion.create(model=self._model, messages=messages, temperature=TEMPERATURE)
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                if attempt == MAX_ATTEMPTS: raise
                logger.warning("LLM request failed (attempt %s/%s): %s", attempt, MAX_ATTEMPTS, exc)
                time.sleep(1.0 * (2 ** (attempt - 1)))
        raise RuntimeError("OpenAI completion failed")

class LocalHTTPChatClient(BaseChatClient):
    def __init__(self, endpoint: str, model: str = OPENAI_MODEL, timeout: float = LOCAL_TIMEOUT):
        if not endpoint or requests is None: raise RuntimeError("Local endpoint requires URL and requests package")
        self._endpoint, self._model, self._timeout = endpoint, model, timeout
        self._log_dir = Path(os.getenv("TEACHER_LLM_LOG_DIR", "/mnt/ssd/logs/teacher_llm"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Using local LLM endpoint=%s timeout=%.1fs", endpoint, timeout)
    def _complete(self, messages):
        payload = {"model": self._model, "messages": messages, "temperature": TEMPERATURE, "stream": False}
        if LEARNING_STAGE == 0: payload["max_tokens"] = TEACHER_STAGE0_MAX_TOKENS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(self._endpoint, json=payload, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    if "choices" in data: return data["choices"][0]["message"]["content"].strip()
                    if "message" in data: return str(data["message"].get("content", "")).strip()
                raise ValueError(f"Unexpected response payload: {data}")
            except Exception as exc:
                if attempt == MAX_ATTEMPTS: raise
                logger.warning("Local LLM request failed (attempt %s/%s): %s", attempt, MAX_ATTEMPTS, exc)
                time.sleep(1.0 * (2 ** (attempt - 1)))
        raise RuntimeError("Local LLM completion failed")

class TeacherAgent:
    def __init__(self, mqtt_client: Optional[mqtt.Client] = None, llm_client: Optional[BaseChatClient] = None, mem_client: Optional[MemRPC] = None):
        self.client = mqtt_client or mqtt.Client(client_id="teacher_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect, self.client.on_message, self.client.on_disconnect = self._on_connect, self._on_message, self._on_disconnect
        self.llm = llm_client
        self.scene: Optional[Dict] = None
        self.snapshot: Optional[str] = None
        self.actions: Deque[str] = deque(maxlen=MAX_HISTORY)
        self._lock = threading.Lock()
        self._inflight = False
        self._last_request = 0.0
        self.mem_store_topic, self.guide_key, self.goal_topic = MEM_STORE_TOPIC, GUIDE_KEY, GOAL_TOPIC
        self.mem_rpc: Optional[MemRPC] = mem_client
        self._backoff_until, self._last_backoff_publish = 0.0, 0.0
        self._last_fingerprint, self._repeat_count = None, 0
        self._fail_streak, self._fail_streak_at_backoff = 0, 0
        self.game_keywords, self.respawn_keywords, self.death_scopes = set(TEACHER_GAME_KEYWORDS), set(TEACHER_RESPAWN_KEYWORDS), set(TEACHER_DEATH_SCOPES or set())
        self.death_scopes.add("death_dialog")
        self._last_scope_wait_reason: Optional[str] = None
        self._context_cache, self._context_text_fingerprint, self._context_refresh_times, self._context_force_refresh = {}, {}, {}, set()
        self._watchdog_last: Dict[str, float] = {}
        self.game_id: Optional[str] = None
        self.identity: Dict[str, object] = {}
        self._last_gate_log = 0.0
        self._gate_held = False

    def _ensure_mem_rpc(self):
        if self.mem_rpc:
            return self.mem_rpc
        try:
            self.mem_rpc = MemRPC(
                host=MQTT_HOST,
                port=MQTT_PORT,
                query_topic=MEM_QUERY_TOPIC,
                reply_topic=MEM_RESPONSE_TOPIC,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemRPC unavailable: %s", exc)
            self.mem_rpc = None
        return self.mem_rpc

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            topics = [(t, 0) for t in [SCENE_TOPIC, SIM_TOPIC, SNAPSHOT_TOPIC, ACT_RESULT_TOPIC, GAME_IDENTITY_TOPIC] if t]
            client.subscribe(topics)
            client.publish("teacher/status", json.dumps({"ok": True, "event": "teacher_ready"}))
            logger.info("Teacher agent connected and subscribed")
        else:
            logger.error("Teacher agent failed to connect: rc=%s", _as_int(rc))
            client.publish("teacher/status", json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))
    
    def _on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0: logger.warning("Teacher agent disconnected: rc=%s", _as_int(rc))

    def _on_message(self, client, _userdata, msg):
        with self._lock:
            try: data = json.loads(msg.payload.decode("utf-8", "ignore"))
            except Exception: data = msg.payload.decode("utf-8", "ignore")
            if msg.topic in {SCENE_TOPIC, SIM_TOPIC} and isinstance(data, dict) and data.get("ok"):
                self.scene = data
                self._handle_scene_update(data)
                if TEACHER_REQUIRE_IN_GAME and not _scene_in_game(data):
                    return
                self._maybe_request_action(client)
            elif msg.topic == SNAPSHOT_TOPIC and isinstance(data, dict) and data.get("ok"): self.snapshot = data.get("image_b64")
            elif msg.topic == ACT_RESULT_TOPIC and isinstance(data, dict) and data.get("applied"): self.actions.append(str(data["applied"]))
            elif msg.topic == GAME_IDENTITY_TOPIC and isinstance(data, dict):
                self._handle_identity_update(data)
    
    def _ensure_llm(self):
        if self.llm: return self.llm
        if LOCAL_ENDPOINT:
            self.llm = LocalHTTPChatClient(LOCAL_ENDPOINT)
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = OpenAIChatClient(os.getenv("OPENAI_API_KEY"))
        return self.llm

    def _handle_scene_update(self, scene: dict):
        if not TEACHER_ALLOW_OCR_GAME_ID:
            return
        if not self.game_id and scene.get("text"):
            text_content = " ".join(scene["text"])
            if len(text_content) > 20:
                thread = threading.Thread(target=self._identify_game, args=(text_content,), daemon=True)
                thread.start()

    def _handle_identity_update(self, payload: dict) -> None:
        self.identity = payload
        raw = payload.get("game_id") or payload.get("app_name") or payload.get("window_title") or payload.get("bundle_id") or ""
        normalized = _normalize_game_id(str(raw))
        if normalized and normalized != "unknown_game":
            if self.game_id != normalized:
                self.game_id = normalized
                logger.info("Identity game_id set: %s (source=%s)", self.game_id, payload.get("source") or "identity")

    def _identify_game(self, text_content):
        acquired = False
        try:
            acquired = acquire_gate("teacher_identify", wait_s=0.0)
            if not acquired:
                return
            llm = self._ensure_llm()
            if llm:
                prompt = [
                    {"role": "system", "content": "Identify the video game from the UI text."},
                    {"role": "user", "content": f"UI: {text_content[:500]}\nName the game and objective."}
                ]
                response = llm._complete(prompt)
                with self._lock:
                    self.game_id = response.split("\n")[0]
                logger.info("Game Identified: %s", self.game_id)
        except Exception as e:
            logger.warning("Identification failed: %s", e)
        finally:
            if acquired:
                release_gate()

    def _scene_scope(self, scene: dict) -> str:
        if scene.get("flags", {}).get("death"):
            return "critical_dialog:death"
        return TEACHER_CONTEXT_SCOPE_FALLBACK

    def _format_scene_desc(self, scene: dict) -> str:
        elements = []
        for target in scene.get("targets", []) or []:
            label = target.get("label", "ui")
            center = target.get("center", [0.5, 0.5])
            elements.append(f'UI "{label}" at ({center[0]:.2f}, {center[1]:.2f})')
        for obj in scene.get("objects", []) or []:
            label = obj.get("label") or obj.get("class") or "obj"
            pos = obj.get("pos")
            if not pos and obj.get("bbox"):
                bbox = obj["bbox"]
                pos = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            pos = pos or [0.5, 0.5]
            elements.append(f'Obj "{label}" at ({pos[0]:.2f}, {pos[1]:.2f})')
        if elements:
            scene_desc = " | ".join(elements)
            raw_text = scene.get("text", [])
            if len(raw_text) < 5:
                scene_desc += f" | Raw Text: {raw_text}"
            return scene_desc
        return json.dumps(scene.get("text", []))

    def _format_prompt_signals(self, scene: dict) -> str:
        scores = scene.get("prompt_scores")
        if not isinstance(scores, dict) or not scores:
            return ""
        items = []
        for label, raw_score in scores.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score < TEACHER_PROMPT_MIN_SCORE:
                continue
            name = str(label).strip().lower()
            if not name:
                continue
            if TEACHER_PROMPT_ALLOW and name not in TEACHER_PROMPT_ALLOW:
                continue
            items.append((name, score))
        if not items:
            return ""
        items.sort(key=lambda pair: pair[1], reverse=True)
        top = items[:max(1, TEACHER_PROMPT_TOP_K)]
        return ", ".join(f"{label}({score:.3f})" for label, score in top)

    def _format_recent_actions(self, actions: List[str]) -> str:
        if not actions:
            return "none"
        return "\n".join(f"- {action}" for action in actions[-5:])

    def _query_mem(self, payload: dict, timeout: float = 1.0) -> Optional[dict]:
        mem = self._ensure_mem_rpc()
        if not mem:
            return None
        try:
            return mem.query(payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem query failed: %s", exc)
            return None

    def _build_scene_summary(self, scene: dict, rules: List[dict], recent_critical: List[dict]) -> str:
        parts = []
        game_id = scene.get("game_id") or self.game_id or CONTROL_PROFILE_GAME_ID
        if game_id:
            parts.append(f"Game: {game_id}")
        if isinstance(self.identity, dict) and self.identity:
            app = self.identity.get("app_name") or ""
            title = self.identity.get("window_title") or ""
            bundle = self.identity.get("bundle_id") or ""
            ident_bits = [item for item in (app, title, bundle) if item]
            if ident_bits:
                parts.append(f"Identity: {' | '.join(str(item) for item in ident_bits[:3])}")
        parts.append(self._format_scene_desc(scene))
        stats = scene.get("stats")
        if stats:
            parts.append(f"Stats: {json.dumps(stats)}")
        prompt_signals = self._format_prompt_signals(scene)
        if prompt_signals:
            parts.append(f"Prompt signals: {prompt_signals}")
        if rules:
            rules_text = "\n".join(f"- {rule.get('text')}" for rule in rules)
            parts.append(f"Rules:\n{rules_text}")
        if recent_critical:
            crit_text = "\n".join(
                f"- {entry.get('delta') or entry.get('event')} ({entry.get('episode_id')})"
                for entry in recent_critical
            )
            parts.append(f"Recent critical:\n{crit_text}")
        if TEACHER_INCLUDE_CONTROLS:
            controls_note = self._format_control_summary(scene)
            if controls_note:
                parts.append(f"Controls: {controls_note}")
        return "\n".join(parts)

    def _format_control_summary(self, scene: dict) -> str:
        game_id = scene.get("game_id") or self.game_id or CONTROL_PROFILE_GAME_ID
        profile = load_profile(str(game_id)) or load_profile("unknown_game") or safe_profile(str(game_id))
        controls = []
        if profile.get("allow_mouse_move", True):
            controls.append("mouse_move")
        if profile.get("allow_primary", True):
            controls.append("click_primary")
        if profile.get("allow_secondary", False):
            controls.append("click_secondary")
        keys = [str(k).lower() for k in profile.get("allowed_keys", []) if k]
        if keys:
            controls.append(f"keys: {', '.join(sorted(keys))}")
        forbidden = [str(k).lower() for k in profile.get("forbidden_keys", []) if k]
        if forbidden:
            controls.append(f"forbidden: {', '.join(sorted(forbidden))}")
        mouse_mode = profile.get("mouse_mode")
        if mouse_mode:
            controls.append(f"mouse_mode={mouse_mode}")
        return "; ".join(controls)

    def _maybe_refresh_context(self, client, llm: BaseChatClient, scene: dict, scope: str) -> Optional[dict]:
        if not hasattr(llm, "describe_environment"):
            return None
        scene_text = " ".join(scene.get("text", [])).strip()
        objects = scene.get("objects") or []
        if not scene_text and not objects:
            return None
        now = time.time()
        last_refresh = self._context_refresh_times.get(scope, 0.0)
        prev_text = self._context_text_fingerprint.get(scope, "")
        text_fingerprint = _text_fingerprint(scene_text)
        should_refresh = not last_refresh or (now - last_refresh) >= TEACHER_CONTEXT_REFRESH_SEC
        if not should_refresh and scene_text and len(scene_text) >= TEACHER_CONTEXT_MIN_CHARS:
            change_ratio = _text_change_ratio(prev_text, text_fingerprint)
            if change_ratio >= TEACHER_CONTEXT_DIFF_JACCARD and len(scene_text) >= TEACHER_CONTEXT_DIFF_MIN_CHARS:
                should_refresh = True
        if not should_refresh:
            return self._context_cache.get(scope)

        object_labels = [str(obj.get("label") or obj.get("class") or "obj") for obj in objects[:TEACHER_CONTEXT_OBJECT_LIMIT]]
        object_summary = ", ".join(object_labels)
        try:
            raw_context = llm.describe_environment(scene_text[:TEACHER_CONTEXT_TEXT_LIMIT], object_summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Context describe failed: %s", exc)
            return None
        try:
            context_payload = json.loads(raw_context) if isinstance(raw_context, str) else raw_context
        except Exception:
            context_payload = {"summary": str(raw_context)}

        if isinstance(context_payload, dict):
            context_payload.setdefault("timestamp", now)
            context_payload.setdefault("scope", scope)
            context_key = f"{TEACHER_CONTEXT_KEY}:{scope}"
            self._context_cache[scope] = context_payload
            self._context_text_fingerprint[scope] = text_fingerprint
            self._context_refresh_times[scope] = now
            client.publish(
                self.mem_store_topic,
                json.dumps({"op": "set", "key": context_key, "value": context_payload, "timestamp": now}),
            )
        return context_payload if isinstance(context_payload, dict) else None

    def _maybe_request_action(self, client):
        now = time.time()
        if self._inflight: return
        if now - self._last_request < REQUEST_INTERVAL: return
        reason = blocked_reason()
        if reason:
            if now - self._last_gate_log >= LLM_GATE_LOG_SEC:
                logger.info("LLM blocked (%s); skipping request", reason)
                self._last_gate_log = now
            return
        if not acquire_gate("teacher_action", wait_s=0.0):
            if now - self._last_gate_log >= LLM_GATE_LOG_SEC:
                logger.info("LLM gate busy; skipping request")
                self._last_gate_log = now
            return
        
        self._gate_held = True
        self._inflight = True
        self._last_request = now
        thread = threading.Thread(target=self._generate_action, args=(client,), daemon=True)
        thread.start()

    def _generate_action(self, client):
        try:
            with self._lock:
                if self.scene is None: return
                scene, snapshot, actions = self.scene, self.snapshot, list(self.actions)
            if TEACHER_REQUIRE_IN_GAME and not _scene_in_game(scene):
                return
            
            llm = self._ensure_llm()
            if llm is None:
                logger.warning("No LLM available")
                return

            scope = self._scene_scope(scene)
            rules = []
            recent_critical = []
            rules_resp = self._query_mem({"mode": "rules", "scope": scope, "limit": TEACHER_RULE_LIMIT})
            if rules_resp and isinstance(rules_resp.get("value"), list):
                rules = rules_resp["value"]
            critical_resp = self._query_mem({"mode": "recent_critical", "scope": scope, "limit": TEACHER_RECENT_CRITICAL_LIMIT})
            if critical_resp and isinstance(critical_resp.get("value"), list):
                recent_critical = critical_resp["value"]

            scene_summary = self._build_scene_summary(scene, rules, recent_critical)
            snapshot_hint = f"{len(snapshot)} bytes" if snapshot else "none"
            recent_actions = self._format_recent_actions(actions)

            context_payload = self._maybe_refresh_context(client, llm, scene, scope)
            context_game = None
            if isinstance(context_payload, dict):
                context_game = context_payload.get("game")

            reasoning = llm.summarize(scene_summary, snapshot_hint, recent_actions)
            action_text = llm.propose_action(reasoning, scene_summary, recent_actions)
            if not action_text:
                return

            payload = {
                "ok": True,
                "timestamp": time.time(),
                "action": action_text,
                "reasoning": reasoning,
                "text": action_text,
                "rules_used": len(rules),
                "recent_critical_used": len(recent_critical),
                "game_id": self.game_id,
                "context_game": context_game,
            }
            client.publish(TEACHER_TOPIC, json.dumps(payload))

        except Exception as exc:
            logger.error("LLM error: %s", exc)
        finally:
            with self._lock:
                self._inflight = False
                if self._gate_held:
                    release_gate()
                    self._gate_held = False

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Teacher agent shut down.")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received, shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = TeacherAgent()
    agent.start()

if __name__ == "__main__":
    main()
