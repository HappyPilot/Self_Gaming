"""Minimal client to query an OpenAI-compatible (e.g., llama.cpp) endpoint for control profiles."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

import requests

from llm_gate import acquire_gate, release_gate

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "8.0"))
LLM_PROFILE_MAX_TOKENS = int(os.getenv("LLM_PROFILE_MAX_TOKENS", "240"))
LLM_GAME_ID_MAX_TOKENS = int(os.getenv("LLM_GAME_ID_MAX_TOKENS", "30"))
LLM_PROMPT_MAX_TOKENS = int(os.getenv("LLM_PROMPT_MAX_TOKENS", "180"))
LLM_DETECT_GAME = os.getenv("LLM_DETECT_GAME", "1") != "0"
_RESOLVED_MODEL: Optional[str] = None


def _models_endpoint() -> str:
    if "/v1/chat/completions" in LLM_ENDPOINT:
        return LLM_ENDPOINT.replace("/v1/chat/completions", "/v1/models")
    if LLM_ENDPOINT.endswith("/v1"):
        return f"{LLM_ENDPOINT}/models"
    return f"{LLM_ENDPOINT.rstrip('/')}/models"


def _resolve_model() -> str:
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    model = (LLM_MODEL or "").strip()
    if model and model.lower() != "auto":
        _RESOLVED_MODEL = model
        return model
    try:
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        resp = requests.get(_models_endpoint(), headers=headers, timeout=min(LLM_TIMEOUT, 5.0))
        if resp.status_code == 200:
            payload = resp.json()
            candidates = payload.get("models") or payload.get("data") or []
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                name = first.get("id") or first.get("name") or first.get("model")
                if name:
                    _RESOLVED_MODEL = str(name)
                    return _RESOLVED_MODEL
    except Exception:
        pass
    _RESOLVED_MODEL = model or "llama"
    return _RESOLVED_MODEL


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract JSON object from free-form text (possibly inside ``` blocks)."""
    if not text:
        return None
    # Grab content inside code fences if present.
    fence_match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1) if fence_match else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
    except Exception:
        pass
    try:
        import ast

        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def fetch_control_profile(game_hint: str, texts: Optional[list] = None) -> Tuple[Optional[dict], str]:
    """Ask LLM for a control profile with descriptions/usage. Returns (profile, status)."""
    gate_acquired = acquire_gate("control_profile")
    if not gate_acquired:
        return None, "llm_gate_busy"
    texts = texts or []
    prompt = (
        "Given a PC game, propose a safe control profile for autonomous agents.\n"
        "Return JSON ONLY with fields:\n"
        "{\n"
        '  "game_id": str,\n'
        '  "mouse_mode": "click_to_move"|"hold_to_move"|"aim_only",\n'
        "  \"allow_mouse_move\": bool,\n"
        "  \"allow_primary\": bool,\n"
        "  \"allow_secondary\": bool,\n"
        "  \"allowed_keys_safe\": [\"w\", \"a\", \"s\", \"d\", \"space\", ...],\n"
        "  \"allowed_keys_extended\": [\"q\", \"e\", \"r\", \"1\", \"2\", \"tab\", ...],\n"
        "  \"forbidden_keys\": [\"alt+f4\", ...],\n"
        "  \"max_actions_per_window\": int,\n"
        "  \"window_sec\": float,\n"
        "  \"notes\": [\"...\"],\n"
        "  \"confidence\": 0.0-1.0\n"
        "}\n"
        "Rules: be conservative; avoid system keys; keep allowed_keys_safe short and game-safe.\n"
        "Only include WASD if the game is known to use WASD by default. For click-to-move ARPGs, prefer QWERT + 1-5.\n"
        "Put inventory/map/menu keys in allowed_keys_extended (not in safe).\n"
        f"Game hint: {game_hint or 'unknown'}.\n"
        f"On-screen texts: {texts[:12]}.\n"
        "JSON only, no prose."
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": _resolve_model(),
        "messages": [
            {"role": "system", "content": "You are a concise assistant that outputs only JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": max(64, LLM_PROFILE_MAX_TOKENS),
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
        if resp.status_code != 200:
            return None, f"llm_http_{resp.status_code}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            return None, "llm_parse_failed"
        return parsed, "llm_ok"
    except Exception as exc:
        return None, f"llm_request_failed: {exc}"
    finally:
        release_gate()


def guess_game_id(game_hint: str, texts: Optional[list] = None) -> Tuple[Optional[str], str]:
    """Ask LLM to guess a short game id/name from visible text."""
    if not LLM_DETECT_GAME:
        return None, "llm_detect_disabled"
    gate_acquired = acquire_gate("game_id")
    if not gate_acquired:
        return None, "llm_gate_busy"
    texts = texts or []
    prompt = (
        "Identify the PC game title from the hints. Reply with only a short lowercase identifier (slug), "
        "no prose. If unsure, reply 'unknown_game'.\n"
        f"Hint: {game_hint or 'unknown'}\n"
        f"On-screen texts: {texts[:15]}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": _resolve_model(),
        "messages": [
            {"role": "system", "content": "Return only the game title or slug."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max(12, LLM_GAME_ID_MAX_TOKENS),
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
        if resp.status_code != 200:
            return None, f"llm_http_{resp.status_code}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Clean minimal slug
        slug = content.strip().strip('"').replace("\n", " ").strip()
        return (slug or None), "llm_ok"
    except Exception as exc:
        return None, f"llm_request_failed: {exc}"
    finally:
        release_gate()


def fetch_visual_prompts(game_hint: str, texts: Optional[list] = None) -> Tuple[Optional[dict], str]:
    """Ask LLM for a compact prompt list to guide zero-shot vision."""
    gate_acquired = acquire_gate("visual_prompts")
    if not gate_acquired:
        return None, "llm_gate_busy"
    texts = texts or []
    prompt = (
        "Suggest a concise list of visual prompt labels for zero-shot detection in a PC game.\n"
        "Return JSON ONLY with fields:\n"
        "{\n"
        '  "game_id": str,\n'
        '  "prompts": [str, ...],\n'
        '  "object_prompts": [str, ...],\n'
        '  "ui_prompts": [str, ...],\n'
        '  "confidence": 0.0-1.0\n'
        "}\n"
        "Rules:\n"
        "- Keep prompts short nouns or noun phrases (1-3 words).\n"
        "- Include generic items: enemy, boss, player, npc, loot, chest, portal, waypoint.\n"
        "- Put UI-only labels in ui_prompts (inventory, map, dialog, menu, loading screen).\n"
        "- Avoid system keys and avoid game-specific spoilers if unsure.\n"
        f"Game hint: {game_hint or 'unknown'}.\n"
        f"On-screen texts: {texts[:12]}.\n"
        "JSON only, no prose."
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": _resolve_model(),
        "messages": [
            {"role": "system", "content": "You output only JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": max(64, LLM_PROMPT_MAX_TOKENS),
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
        if resp.status_code != 200:
            return None, f"llm_http_{resp.status_code}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            return None, "llm_parse_failed"
        return parsed, "llm_ok"
    except Exception as exc:
        return None, f"llm_request_failed: {exc}"
    finally:
        release_gate()
