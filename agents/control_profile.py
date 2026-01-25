"""Shared helpers for per-game control profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROFILE_PATH = Path(os.getenv("CONTROL_PROFILE_PATH", "/app/data/control_profiles.json"))
EXAMPLE_PROFILE_PATH = Path(__file__).resolve().parent / "data" / "control_profiles.example.json"


def _read_all_profiles(path: Path = DEFAULT_PROFILE_PATH) -> Dict[str, dict]:
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
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            profiles.update(data)
            return profiles
    except Exception:
        return profiles
    return profiles


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
        "allowed_keys": ["w", "a", "s", "d", "q", "e", "r", "t", "1", "2", "3", "4", "5", "space"],
        "allowed_keys_extended": [],
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
