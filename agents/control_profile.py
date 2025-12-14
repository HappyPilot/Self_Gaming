"""Shared helpers for per-game control profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROFILE_PATH = Path(os.getenv("CONTROL_PROFILE_PATH", "/app/data/control_profiles.json"))


def _read_all_profiles(path: Path = DEFAULT_PROFILE_PATH) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _write_all_profiles(profiles: Dict[str, dict], path: Path = DEFAULT_PROFILE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def safe_profile(game_id: str = "unknown_game") -> dict:
    """Return a conservative profile that avoids risky keys."""
    return {
        "game_id": game_id,
        "source": "safe_default",
        "profile_version": 1,
        "mouse_mode": "click_to_move",
        "allow_mouse_move": True,
        "allow_primary": True,
        "allow_secondary": False,
        "allowed_keys": [],
        "forbidden_keys": [],
        "max_actions_per_window": 6,
        "window_sec": 10.0,
        "notes": ["Generated safe fallback profile"],
    }


def load_profile(game_id: str, path: Path = DEFAULT_PROFILE_PATH) -> Optional[dict]:
    profiles = _read_all_profiles(path)
    return profiles.get(game_id)


def upsert_profile(profile: dict, path: Path = DEFAULT_PROFILE_PATH) -> None:
    if not isinstance(profile, dict):
        return
    game_id = profile.get("game_id") or "unknown_game"
    profiles = _read_all_profiles(path)
    profiles[str(game_id)] = profile
    _write_all_profiles(profiles, path)
