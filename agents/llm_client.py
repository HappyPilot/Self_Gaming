"""Minimal client to query an OpenAI-compatible (e.g., llama.cpp) endpoint for control profiles."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

import requests

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "8.0"))
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
        return None


def fetch_control_profile(game_hint: str, texts: Optional[list] = None) -> Tuple[Optional[dict], str]:
    """Ask LLM for a control profile with descriptions/usage. Returns (profile, status)."""
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
        "  \"controls\": [\n"
        "    {\"input\": \"key_w\", \"description\": \"move forward\", \"when_to_use\": \"movement\", \"risk_level\": \"low\", \"required\": true, \"safe_to_probe\": true},\n"
        "    ...\n"
        "  ],\n"
        "  \"forbidden_keys\": [\"alt+f4\", ...],\n"
        "  \"max_actions_per_window\": int,\n"
        "  \"window_sec\": float,\n"
        "  \"notes\": [\"...\"],\n"
        "  \"confidence\": 0.0-1.0\n"
        "}\n"
        "Rules: be conservative; describe purpose and context in `description` and `when_to_use`; mark risky/system keys as forbidden; set `safe_to_probe=false` for anything that can open menus or exit.\n"
        f"Game hint: {game_hint or 'unknown'}.\n"
        f"On-screen texts: {texts[:15]}.\n"
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
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
    except Exception as exc:
        return None, f"llm_request_failed: {exc}"
    if resp.status_code != 200:
        return None, f"llm_http_{resp.status_code}"
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return None, "llm_bad_response"
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        return None, "llm_parse_failed"
    return parsed, "llm_ok"


def guess_game_id(game_hint: str, texts: Optional[list] = None) -> Tuple[Optional[str], str]:
    """Ask LLM to guess a short game id/name from visible text."""
    if not LLM_DETECT_GAME:
        return None, "llm_detect_disabled"
    texts = texts or []
    prompt = (
        "Identify the PC game title from the hints. Reply with only a short identifier (slug or name), "
        "no prose.\n"
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
        "max_tokens": 30,
    }
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=LLM_TIMEOUT)
    except Exception as exc:
        return None, f"llm_request_failed: {exc}"
    if resp.status_code != 200:
        return None, f"llm_http_{resp.status_code}"
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None, "llm_bad_response"
    # Clean minimal slug
    slug = content.strip().strip('"').replace("\n", " ").strip()
    return (slug or None), "llm_ok"
