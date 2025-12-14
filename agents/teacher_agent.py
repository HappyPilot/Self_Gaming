#!/usr/bin/env python3
"""Teacher agent that queries an LLM for high-level guidance."""
from __future__ import annotations

import json
import logging
import os
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

# --- Setup ---
logging.basicConfig(level=os.getenv("TEACHER_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("teacher_agent")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST, MQTT_PORT = os.getenv("MQTT_HOST", "127.0.0.1"), int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC, SIM_TOPIC = os.getenv("SCENE_TOPIC", "scene/state"), os.getenv("SIM_TOPIC", "sim_core/state")
SNAPSHOT_TOPIC, ACT_RESULT_TOPIC = os.getenv("SNAPSHOT_TOPIC", "vision/snapshot"), os.getenv("ACT_RESULT_TOPIC", "act/result")
TEACHER_TOPIC, MEM_STORE_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action"), os.getenv("MEM_STORE_TOPIC", "mem/store")
MEM_QUERY_TOPIC, MEM_REPLY_TOPIC, MEM_RESPONSE_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query"), os.getenv("MEM_REPLY_TOPIC", "mem/reply"), os.getenv("MEM_RESPONSE_TOPIC", "mem/response")
GOAL_TOPIC, GUIDE_KEY = os.getenv("GOAL_TOPIC", "goals/high_level"), os.getenv("TEACHER_GUIDE_KEY", "teacher_guides")
TEACHER_PROVIDER, OPENAI_MODEL = os.getenv("TEACHER_PROVIDER", "openai").lower(), os.getenv("TEACHER_OPENAI_MODEL", "gpt-4o-mini")
LOCAL_ENDPOINT, LOCAL_TIMEOUT = os.getenv("TEACHER_LOCAL_ENDPOINT"), float(os.getenv("TEACHER_LOCAL_TIMEOUT", "20"))
MAX_HISTORY, REQUEST_INTERVAL = int(os.getenv("TEACHER_HISTORY", "6")), float(os.getenv("TEACHER_INTERVAL_SEC", "2.0"))
MAX_ATTEMPTS, TEMPERATURE = int(os.getenv("TEACHER_MAX_ATTEMPTS", "3")), float(os.getenv("TEACHER_TEMPERATURE", "0.2"))
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

class BaseChatClient:
    def summarize(self, scene_summary: str, snapshot_hint: str, recent_actions: str) -> str: raise NotImplementedError
    def propose_action(self, reasoning: str, scene_summary: str, recent_actions: str) -> str: raise NotImplementedError
    def describe_environment(self, scene_text: str, objects: str) -> str: raise NotImplementedError

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

    def _on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            topics = [(t, 0) for t in [SCENE_TOPIC, SIM_TOPIC, SNAPSHOT_TOPIC, ACT_RESULT_TOPIC] if t]
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
                self._maybe_request_action(client)
            elif msg.topic == SNAPSHOT_TOPIC and isinstance(data, dict) and data.get("ok"): self.snapshot = data.get("image_b64")
            elif msg.topic == ACT_RESULT_TOPIC and isinstance(data, dict) and data.get("applied"): self.actions.append(str(data["applied"]))
    
    def _ensure_llm(self):
        if self.llm: return self.llm
        if LOCAL_ENDPOINT:
            self.llm = LocalHTTPChatClient(LOCAL_ENDPOINT)
        elif os.getenv("OPENAI_API_KEY"):
            self.llm = OpenAIChatClient(os.getenv("OPENAI_API_KEY"))
        return self.llm

    def _handle_scene_update(self, scene: dict):
        if not self.game_id and scene.get("text"):
            text_content = " ".join(scene["text"])
            if len(text_content) > 20:
                thread = threading.Thread(target=self._identify_game, args=(text_content,), daemon=True)
                thread.start()

    def _identify_game(self, text_content):
        try:
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

    def _maybe_request_action(self, client):
        now = time.time()
        if self._inflight: return
        if now - self._last_request < REQUEST_INTERVAL: return
        
        self._inflight = True
        self._last_request = now
        thread = threading.Thread(target=self._generate_action, args=(client,), daemon=True)
        thread.start()

    def _generate_action(self, client):
        try:
            with self._lock:
                if self.scene is None: return
                scene, snapshot, actions = self.scene, self.snapshot, list(self.actions)
            
            llm = self._ensure_llm()
            if llm is None:
                logger.warning("No LLM available")
                return
            
            # Construct spatial map for the Teacher
            elements = []
            # 1. UI Targets (buttons, text zones with coords)
            for t in scene.get('targets', []):
                label = t.get('label', 'ui')
                center = t.get('center', [0.5, 0.5])
                # Format: UI "Save" at (0.2, 0.8)
                elements.append(f'UI "{label}" at ({center[0]:.2f}, {center[1]:.2f})')
            
            # 2. Game Objects (enemies, player, loot)
            for o in scene.get('objects', []):
                label = o.get('label') or o.get('class') or 'obj'
                # Some objects store pos as 'pos', others might calculate it from bbox
                pos = o.get('pos')
                if not pos and o.get('bbox'):
                    b = o['bbox']
                    pos = [(b[0]+b[2])/2, (b[1]+b[3])/2]
                pos = pos or [0.5, 0.5]
                elements.append(f'Obj "{label}" at ({pos[0]:.2f}, {pos[1]:.2f})')

            # 3. Fallback or Append Raw Text
            if not elements:
                scene_desc = json.dumps(scene.get('text', []))
            else:
                scene_desc = " | ".join(elements)
                raw_text = scene.get('text', [])
                if len(raw_text) < 5:
                    scene_desc += f" | Raw Text: {raw_text}"

            # History for context
            history_str = "\n".join([f"- {a}" for a in list(self.actions)[-5:]])

            prompt = [
                {"role": "system", "content": f"You are an expert player of {self.game_id or 'this game'}. {TARGET_HINT_INSTRUCTIONS}\n\nCRITICAL: If previous actions failed to move the character or change the scene, try a DIFFERENT approach (e.g., hold mouse, click elsewhere, use keyboard)."},
                {"role": "user", "content": f"Scene: {scene_desc}\nStatus: {json.dumps(scene.get('stats', {}))}\nRecent Actions:\n{history_str}\n\nProblem: Character might be stuck. What exactly should I do?"}
            ]
            
            response = llm._complete(prompt)
            logger.info("Teacher says: %s", response)

            # --- THOUGHT LOG ---
            try:
                log_dir = "/app/logs"
                if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
                with open(f"{log_dir}/thought_process.log", "a") as f:
                    entry = {
                        "timestamp": time.time(),
                        "game": self.game_id,
                        "scene": scene_desc[:300],
                        "history": list(self.actions)[-3:],
                        "advice": response
                    }
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to write thought log: {e}")
            # -------------------
            
            payload = {"ok": True, "text": response, "timestamp": time.time(), "game_id": self.game_id}
            client.publish(TEACHER_TOPIC, json.dumps(payload))
            
        except Exception as exc:
            logger.error("LLM error: %s", exc)
        finally:
            with self._lock: self._inflight = False

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