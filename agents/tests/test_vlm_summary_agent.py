from agents import vlm_summary_agent as vsa


def test_parse_vlm_summary_payload():
    payload = {
        "game": "path_of_exile",
        "summary": "town hub, NPCs nearby",
        "player_state": "healthy",
        "enemies": "none",
        "objectives": "find stash",
        "ui": "inventory closed",
        "risk": "low",
        "recommended_intent": "move to stash",
        "timestamp": 123.0,
    }
    parsed = vsa.normalize_summary(payload)
    assert parsed["game"] == "path_of_exile"
    assert parsed["risk"] == "low"
    assert "summary" in parsed
