"""Helpers for control profile v2 bindings and grouping."""
from __future__ import annotations

AUX_CATEGORIES = {"ui", "social", "system"}
GAMEPLAY_CATEGORIES = {"movement", "combat", "interaction"}


def _split_keys_and_combos(values: list[str]) -> tuple[list[str], list[str]]:
    keys = []
    combos = []
    for value in values:
        if "+" in value:
            combos.append(value)
        else:
            keys.append(value)
    return keys, combos


def normalize_profile_v2(profile: dict) -> dict:
    profile = dict(profile or {})
    bindings = profile.get("bindings") or []
    allowed = set()
    blocked = set()
    allowed_combos = set()
    blocked_combos = set()
    for binding in bindings:
        category = str(binding.get("category") or "").lower()
        keys = [str(k).lower() for k in (binding.get("keys") or []) if k]
        plain_keys, combo_keys = _split_keys_and_combos(keys)
        if category in AUX_CATEGORIES:
            blocked.update(plain_keys)
            blocked_combos.update(combo_keys)
        elif category in GAMEPLAY_CATEGORIES:
            allowed.update(plain_keys)
            allowed_combos.update(combo_keys)
    profile["allowed_keys_gameplay"] = sorted(allowed)
    profile["blocked_keys_aux"] = sorted(blocked)
    profile["allowed_combos_gameplay"] = sorted(allowed_combos)
    profile["blocked_combos_aux"] = sorted(blocked_combos)
    return profile


def apply_profile_v2(profile: dict) -> dict:
    profile = normalize_profile_v2(profile)
    if not profile.get("allowed_keys"):
        profile["allowed_keys"] = list(profile.get("allowed_keys_gameplay") or [])
    return profile
