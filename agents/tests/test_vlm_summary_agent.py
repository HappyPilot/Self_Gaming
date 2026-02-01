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


def test_build_messages_without_image_uses_text_context():
    messages = vsa.build_messages(None, "ui text", "enemy")
    assert isinstance(messages[1]["content"], str)
    assert "Scene text" in messages[1]["content"]


def test_normalize_summary_handles_text():
    parsed = vsa.normalize_summary("hello")
    assert parsed["summary"] == "hello"
