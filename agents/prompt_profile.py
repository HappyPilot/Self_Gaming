"""Helpers for per-game vision prompt profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROMPT_PROFILE_PATH = Path(os.getenv("PROMPT_PROFILE_PATH", "/app/data/prompt_profiles.json"))
EXAMPLE_PROFILE_PATH = Path(__file__).resolve().parent / "data" / "prompt_profiles.example.json"


DEFAULT_PROMPTS = [
    "enemy",
    "boss",
    "player",
    "npc",
    "loot",
    "chest",
    "portal",
    "waypoint",
    "objective marker",
    "dialog",
    "minimap",
    "map overlay",
    "inventory",
    "health bar",
    "mana bar",
    "skill bar",
    "death screen",
    "loading screen",
    "pause menu",
]

DEFAULT_OBJECT_PROMPTS = [
    "enemy",
    "boss",
    "player",
    "npc",
    "loot",
    "chest",
    "portal",
    "waypoint",
    "objective marker",
]

DEFAULT_UI_PROMPTS = [
    "dialog",
    "minimap",
    "map overlay",
    "inventory",
    "health bar",
    "mana bar",
    "skill bar",
    "death screen",
    "loading screen",
    "pause menu",
]


def _read_all_profiles(path: Path = DEFAULT_PROMPT_PROFILE_PATH) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    if EXAMPLE_PROFILE_PATH.exists():
        try:
            data = json.loads(EXAMPLE_PROFILE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                profiles.update(data)
        except Exception:
            pass
    if not path.exists():
        return profiles
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            profiles.update(data)
    except Exception:
        return profiles
    return profiles


def _write_all_profiles(profiles: Dict[str, dict], path: Path = DEFAULT_PROMPT_PROFILE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(profiles, handle, ensure_ascii=False, indent=2)
    except Exception:
        return


def fallback_prompt_profile(game_id: str = "unknown_game") -> dict:
    return {
        "game_id": game_id,
        "source": "fallback",
        "prompts": list(DEFAULT_PROMPTS),
        "object_prompts": list(DEFAULT_OBJECT_PROMPTS),
        "ui_prompts": list(DEFAULT_UI_PROMPTS),
        "confidence": 0.0,
    }


def load_prompt_profile(game_id: str, path: Path = DEFAULT_PROMPT_PROFILE_PATH) -> Optional[dict]:
    profiles = _read_all_profiles(path)
    if game_id in profiles:
        return profiles[game_id]
    return profiles.get("generic")


def upsert_prompt_profile(profile: dict, path: Path = DEFAULT_PROMPT_PROFILE_PATH) -> None:
    if not isinstance(profile, dict):
        return
    game_id = profile.get("game_id") or "unknown_game"
    profiles = _read_all_profiles(path)
    profiles[str(game_id)] = profile
    _write_all_profiles(profiles, path)
