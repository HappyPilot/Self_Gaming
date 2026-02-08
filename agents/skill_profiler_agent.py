#!/usr/bin/env python3
"""Skill profiler agent (OCR tooltip -> skill type)."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Optional

import paho.mqtt.client as mqtt
import requests

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SKILL_CMD_TOPIC = os.getenv("SKILL_PROFILE_CMD_TOPIC", "skill_profiler/cmd")
SKILL_RESULT_TOPIC = os.getenv("SKILL_PROFILE_RESULT_TOPIC", "skill_profiler/result")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")
GAME_SCHEMA_TOPIC = os.getenv("GAME_SCHEMA_TOPIC", "game/schema")
DEFAULT_GAME_ID = os.getenv("GAME_ID", "unknown_game")

SKILL_USE_LLM = os.getenv("SKILL_PROFILE_USE_LLM", "0") != "0"
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "8.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_PROFILE_MAX_TOKENS", "64"))

skill_map: Dict[str, str] = {}


def classify_skill_type(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("area", "around", "aoe", "radius", "all enemies", "cone")):
        return "aoe"
    if any(token in lowered for token in ("dash", "blink", "teleport", "leap", "jump", "charge", "movement")):
        return "mobility"
    if any(token in lowered for token in ("shield", "block", "barrier", "guard", "heal", "regen", "invulnerable", "defense")):
        return "defense"
    return "single"


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _classify_with_llm(text: str) -> Optional[str]:
    prompt = (
        "Classify the skill tooltip into one of: aoe, single, mobility, defense.\n"
        "Reply with JSON only: {\"type\": \"aoe|single|mobility|defense\"}.\n"
        f"Tooltip: {text}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max(16, LLM_MAX_TOKENS),
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if isinstance(parsed, dict):
            value = str(parsed.get("type") or "").strip().lower()
            if value in {"aoe", "single", "mobility", "defense"}:
                return value
    except Exception:
        return None
    return None


def _publish(client: mqtt.Client, topic: str, payload: dict) -> None:
    if not topic:
        return
    client.publish(topic, json.dumps(payload))


def _handle_command(client: mqtt.Client, payload: dict) -> None:
    slot = str(payload.get("slot") or payload.get("key") or "").lower()
    text = payload.get("tooltip") or payload.get("text") or ""
    game_id = str(payload.get("game_id") or DEFAULT_GAME_ID)
    if not slot or not text:
        return

    skill_type = classify_skill_type(text)
    if SKILL_USE_LLM:
        llm_value = _classify_with_llm(text)
        if llm_value:
            skill_type = llm_value

    skill_map[slot] = skill_type
    result = {
        "ok": True,
        "event": "skill_profile",
        "game_id": game_id,
        "slot": slot,
        "skill_type": skill_type,
        "timestamp": time.time(),
    }
    _publish(client, SKILL_RESULT_TOPIC, result)
    if MEM_STORE_TOPIC:
        _publish(
            client,
            MEM_STORE_TOPIC,
            {"op": "set", "key": f"skill_profile:{game_id}", "value": dict(skill_map)},
        )
    if GAME_SCHEMA_TOPIC:
        _publish(
            client,
            GAME_SCHEMA_TOPIC,
            {
                "ok": True,
                "schema_patch": {"profile": {"skill_map": dict(skill_map)}},
                "timestamp": time.time(),
            },
        )


def _on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        client.subscribe([(SKILL_CMD_TOPIC, 0)])
    else:
        raise RuntimeError(f"MQTT connect failed: {rc}")


def _on_message(client, _userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8", "ignore"))
    except Exception:
        return
    if isinstance(data, dict):
        _handle_command(client, data)


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="skill_profiler")
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
