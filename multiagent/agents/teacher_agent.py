#!/usr/bin/env python3
"""Teacher agent that queries an LLM for high-level guidance.

The agent listens to the latest fused scene state and optional snapshots, keeps
track of recent actions, and asks an OpenAI Chat Completion model for the next
recommended step. Results are published to the ``teacher/action`` MQTT topic so
other agents (policy, recorder, trainer) can leverage the guidance.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from typing import Deque, Dict, Optional

try:  # Optional dependency for local provider
    import requests
except ImportError:  # pragma: no cover - handled during runtime configuration
    requests = None

import paho.mqtt.client as mqtt

logger = logging.getLogger("teacher_agent")
logging.basicConfig(level=os.getenv("TEACHER_LOG_LEVEL", "INFO"))

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
SIM_TOPIC = os.getenv("SIM_TOPIC", "sim_core/state")
SNAPSHOT_TOPIC = os.getenv("SNAPSHOT_TOPIC", "vision/snapshot")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
TEACHER_TOPIC = os.getenv("TEACHER_ACTION_TOPIC", "teacher/action")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
GOAL_TOPIC = os.getenv("GOAL_TOPIC", "goals/high_level")
GUIDE_KEY = os.getenv("TEACHER_GUIDE_KEY", "teacher_guides")
TEACHER_PROVIDER = os.getenv("TEACHER_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("TEACHER_OPENAI_MODEL", "gpt-4o-mini")
LOCAL_ENDPOINT = os.getenv("TEACHER_LOCAL_ENDPOINT")
LOCAL_TIMEOUT = float(os.getenv("TEACHER_LOCAL_TIMEOUT", "15"))
MAX_HISTORY = int(os.getenv("TEACHER_HISTORY", "6"))
REQUEST_INTERVAL = float(os.getenv("TEACHER_INTERVAL_SEC", "2.0"))
MAX_ATTEMPTS = int(os.getenv("TEACHER_MAX_ATTEMPTS", "3"))
TEMPERATURE = float(os.getenv("TEACHER_TEMPERATURE", "0.2"))


class BaseChatClient:
    """Abstract chat client interface."""

    def summarize(self, scene_summary: str, snapshot_hint: str, recent_actions: str) -> str:  # pragma: no cover - override in subclasses
        raise NotImplementedError

    def propose_action(self, reasoning: str, scene_summary: str, recent_actions: str) -> str:  # pragma: no cover - override in subclasses
        raise NotImplementedError


class OpenAIChatClient(BaseChatClient):
    """Wrapper around the OpenAI ChatCompletion API with retry/backoff."""

    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for teacher agent")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - handled via tests
            raise RuntimeError("openai package is not installed") from exc
        self._openai = openai
        self._openai.api_key = api_key
        self._model = model

    def _complete(self, messages):
        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._openai.ChatCompletion.create(
                    model=self._model,
                    messages=messages,
                    temperature=TEMPERATURE,
                )
                choice = response["choices"][0]["message"]["content"].strip()
                return choice
            except Exception as exc:  # pragma: no cover - network errors
                if attempt == MAX_ATTEMPTS:
                    raise
                logger.warning("LLM request failed (attempt %s/%s): %s", attempt, MAX_ATTEMPTS, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("OpenAI completion failed")

    def summarize(self, scene_summary: str, snapshot_hint: str, recent_actions: str) -> str:
        summary_prompt = (
            "You are observing a UI automation task. Summarize the current scene "
            "and think through the objective before proposing an action.\n"
            f"Scene text: {scene_summary}\n"
            f"Recent actions: {recent_actions or 'none'}\n"
            f"Snapshot (base64 preview): {snapshot_hint or 'n/a'}\n"
            "Provide a concise reasoning paragraph."
        )
        messages = [
            {"role": "system", "content": "You are an expert UI automation teacher."},
            {"role": "user", "content": summary_prompt},
        ]
        return self._complete(messages)

    def propose_action(self, reasoning: str, scene_summary: str, recent_actions: str) -> str:
        action_prompt = (
            "Based on your reasoning, provide the single next high-level action the agent "
            "should take. Respond with an imperative sentence (e.g., 'Press Enter').\n"
            f"Reasoning: {reasoning}\n"
            f"Scene text: {scene_summary}\n"
            f"Recent actions: {recent_actions or 'none'}"
        )
        messages = [
            {"role": "system", "content": "You teach agents how to operate desktop UIs."},
            {"role": "user", "content": action_prompt},
        ]
        return self._complete(messages)


class LocalHTTPChatClient(BaseChatClient):
    """HTTP client for locally hosted LLMs using an OpenAI-compatible schema."""

    def __init__(self, endpoint: str, model: str = OPENAI_MODEL, timeout: float = LOCAL_TIMEOUT):
        if not endpoint:
            raise ValueError("TEACHER_LOCAL_ENDPOINT must be provided for local provider")
        if requests is None:  # pragma: no cover - configuration issue
            raise RuntimeError("requests package is required for local LLM provider")
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout

    def _complete(self, messages):
        delay = 1.0
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "stream": False,
        }
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "choices" in data:
                    choice = data["choices"][0]["message"]["content"].strip()
                    return choice
                if isinstance(data, dict) and "message" in data:
                    content = data["message"].get("content", "")
                    return str(content).strip()
                raise ValueError(f"Unexpected response payload: {data}")
            except Exception as exc:  # pragma: no cover - network errors
                if attempt == MAX_ATTEMPTS:
                    raise
                logger.warning("Local LLM request failed (attempt %s/%s): %s", attempt, MAX_ATTEMPTS, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("Local LLM completion failed")

    def summarize(self, scene_summary: str, snapshot_hint: str, recent_actions: str) -> str:
        summary_prompt = (
            "You are observing a UI automation task. Summarize the current scene "
            "and think through the objective before proposing an action.\n"
            f"Scene text: {scene_summary}\n"
            f"Recent actions: {recent_actions or 'none'}\n"
            f"Snapshot (base64 preview): {snapshot_hint or 'n/a'}\n"
            "Provide a concise reasoning paragraph."
        )
        messages = [
            {"role": "system", "content": "You are an expert UI automation teacher."},
            {"role": "user", "content": summary_prompt},
        ]
        return self._complete(messages)

    def propose_action(self, reasoning: str, scene_summary: str, recent_actions: str) -> str:
        action_prompt = (
            "Based on your reasoning, provide the single next high-level action the agent "
            "should take. Respond with an imperative sentence (e.g., 'Press Enter').\n"
            f"Reasoning: {reasoning}\n"
            f"Scene text: {scene_summary}\n"
            f"Recent actions: {recent_actions or 'none'}"
        )
        messages = [
            {"role": "system", "content": "You teach agents how to operate desktop UIs."},
            {"role": "user", "content": action_prompt},
        ]
        return self._complete(messages)


def _infer_goal_type(text: str) -> str:
    text = text.lower()
    if any(word in text for word in ("farm", "kill", "enemy")):
        return "farm"
    if any(word in text for word in ("loot", "collect", "pick")):
        return "loot"
    if any(word in text for word in ("travel", "move", "explore")):
        return "explore"
    return "task"


class TeacherAgent:
    """Coordinates MQTT I/O and LLM interactions to emit teacher actions."""

    def __init__(self, mqtt_client: Optional[mqtt.Client] = None, llm_client: Optional[OpenAIChatClient] = None):
        self.client = mqtt_client or mqtt.Client(client_id="teacher_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.llm = llm_client
        self.scene: Optional[Dict] = None
        self.snapshot: Optional[str] = None
        self.actions: Deque[str] = deque(maxlen=MAX_HISTORY)
        self._lock = threading.Lock()
        self._inflight = False
        self._last_request = 0.0
        self.mem_store_topic = MEM_STORE_TOPIC
        self.guide_key = GUIDE_KEY
        self.goal_topic = GOAL_TOPIC

    # ------------------------------------------------------------------ MQTT
    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = []
            if SCENE_TOPIC:
                topics.append((SCENE_TOPIC, 0))
            if SIM_TOPIC:
                topics.append((SIM_TOPIC, 0))
            if SNAPSHOT_TOPIC:
                topics.append((SNAPSHOT_TOPIC, 0))
            if ACT_RESULT_TOPIC:
                topics.append((ACT_RESULT_TOPIC, 0))
            client.subscribe(topics)
            client.publish(
                "teacher/status",
                json.dumps({"ok": True, "event": "teacher_ready"}),
            )
            logger.info("Teacher agent connected and subscribed to topics")
        else:
            client.publish(
                "teacher/status",
                json.dumps({"ok": False, "event": "connect_failed", "code": int(rc)}),
            )
            logger.error("Teacher agent failed to connect: rc=%s", rc)

    def _on_disconnect(self, _client, _userdata, rc):
        logger.warning("Teacher agent disconnected: rc=%s", rc)

    def _on_message(self, client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = payload

        if msg.topic in {SCENE_TOPIC, SIM_TOPIC} and isinstance(data, dict) and data.get("ok"):
            self.scene = data
            self._maybe_request_action(client)
        elif msg.topic == SNAPSHOT_TOPIC and isinstance(data, dict) and data.get("ok"):
            self.snapshot = data.get("image_b64")
        elif msg.topic == ACT_RESULT_TOPIC:
            action = data.get("applied") if isinstance(data, dict) else None
            if action:
                self.actions.append(str(action))

    # ----------------------------------------------------------------- LLM IO
    def _ensure_llm(self) -> Optional[BaseChatClient]:
        if self.llm is None:
            provider = TEACHER_PROVIDER
            try:
                if provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        logger.error("OPENAI_API_KEY not set; teacher agent idle")
                        return None
                    self.llm = OpenAIChatClient(api_key)
                elif provider == "local":
                    if not LOCAL_ENDPOINT:
                        logger.error("TEACHER_LOCAL_ENDPOINT not set; cannot use local provider")
                        return None
                    self.llm = LocalHTTPChatClient(LOCAL_ENDPOINT)
                else:
                    logger.error("Unknown TEACHER_PROVIDER '%s'", provider)
                    return None
            except Exception as exc:  # pragma: no cover - config errors
                logger.error("Failed to initialise teacher client: %s", exc)
                self.llm = None
        return self.llm

    def _maybe_request_action(self, client):
        now = time.time()
        if now - self._last_request < REQUEST_INTERVAL:
            return
        if self.scene is None:
            return
        with self._lock:
            if self._inflight:
                return
            self._inflight = True
        threading.Thread(target=self._generate_action, args=(client,), daemon=True).start()

    def _generate_action(self, client):
        try:
            llm = self._ensure_llm()
            if llm is None:
                return
            scene_text = " ".join(self.scene.get("text") or []) if self.scene else ""
            mean = self.scene.get("mean") if self.scene else None
            scene_summary = f"mean={mean}, text={scene_text}" if mean is not None else scene_text
            snapshot_hint = (self.snapshot or "")[:96]
            recent_actions = ", ".join(self.actions)
            reasoning = llm.summarize(scene_summary, snapshot_hint, recent_actions)
            action_text = llm.propose_action(reasoning, scene_summary, recent_actions)
            payload = {
                "ok": True,
                "action": action_text.strip(),
                "reasoning": reasoning.strip(),
                "scene_mean": mean,
                "timestamp": time.time(),
            }
            client.publish(TEACHER_TOPIC, json.dumps(payload))
            self._store_guide(payload)
            self._publish_goal(payload)
            logger.info("Published teacher action: %s", payload["action"])
            self._last_request = time.time()
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.error("Teacher agent failed to generate action: %s", exc)
        finally:
            with self._lock:
                self._inflight = False

    def _store_guide(self, payload: Dict):
        if not self.mem_store_topic:
            return
        guide = {
            "action": payload.get("action"),
            "reasoning": payload.get("reasoning"),
            "timestamp": payload.get("timestamp"),
        }
        message = {"op": "vector_append", "key": self.guide_key, "value": guide}
        self.client.publish(self.mem_store_topic, json.dumps(message))

    def _publish_goal(self, payload: Dict):
        if not self.goal_topic:
            return
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        action_text = payload.get("action", "")
        reasoning = payload.get("reasoning", "")
        goal_type = _infer_goal_type(f"{action_text} {reasoning}")
        goal = {
            "ok": True,
            "goal_id": goal_id,
            "goal_type": goal_type,
            "reasoning": reasoning,
            "action_hint": action_text,
            "timestamp": payload.get("timestamp", time.time()),
        }
        self.client.publish(self.goal_topic, json.dumps(goal))

    # ----------------------------------------------------------------- Public
    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_forever()


def main():
    agent = TeacherAgent()
    agent.start()


if __name__ == "__main__":
    main()
