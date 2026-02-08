from __future__ import annotations

from pathlib import Path


def _load_defaults() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "config" / "defaults.env").read_text(encoding="utf-8")


def _get_value(payload: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in payload.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def test_zeroshot_prompts_are_pruned():
    data = _load_defaults()
    raw = _get_value(data, "ZSD_PROMPTS")
    assert raw, "ZSD_PROMPTS missing from defaults.env"
    prompts = {p.strip() for p in raw.split(",") if p.strip()}
    banned = {
        "minimap",
        "inventory",
        "quest_marker",
        "dialog_button",
        "health_orb",
        "mana_orb",
        "menu_button",
        "portal",
        "waypoint",
    }
    assert not (prompts & banned), f"ZSD_PROMPTS still includes UI tokens: {sorted(prompts & banned)}"


def test_zeroshot_conf_threshold_is_tightened():
    data = _load_defaults()
    raw = _get_value(data, "ZSD_CONF_THRESHOLD")
    assert raw, "ZSD_CONF_THRESHOLD missing from defaults.env"
    assert float(raw) >= 0.1, f"ZSD_CONF_THRESHOLD too low: {raw}"
