import base64
import importlib
import io

import pytest


def _load_vsa(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_SUMMARY_LOG_DIR", str(tmp_path))
    import agents.vlm_summary_agent as vsa

    return importlib.reload(vsa)


def test_parse_vlm_summary_payload(monkeypatch, tmp_path):
    vsa = _load_vsa(monkeypatch, tmp_path)
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


def test_build_messages_without_image_uses_text_context(monkeypatch, tmp_path):
    vsa = _load_vsa(monkeypatch, tmp_path)
    messages = vsa.build_messages(None, None, "ui text", "enemy")
    assert isinstance(messages[1]["content"], str)
    assert "Scene text" in messages[1]["content"]


def test_normalize_summary_handles_text(monkeypatch, tmp_path):
    vsa = _load_vsa(monkeypatch, tmp_path)
    parsed = vsa.normalize_summary("hello")
    assert parsed["summary"] == "hello"


def test_prepare_image_b64_resizes_and_sets_mime(monkeypatch, tmp_path):
    vsa = _load_vsa(monkeypatch, tmp_path)
    pytest.importorskip("PIL")
    from PIL import Image

    monkeypatch.setattr(vsa, "VLM_IMAGE_MAX_DIM", 320, raising=False)
    monkeypatch.setattr(vsa, "VLM_IMAGE_FORMAT", "PNG", raising=False)

    img = Image.new("RGB", (800, 600), (1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    payload = {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii")}

    image_b64, mime = vsa.prepare_image_b64(payload)

    assert mime == "image/png"
    decoded = base64.b64decode(image_b64)
    resized = Image.open(io.BytesIO(decoded))
    assert max(resized.size) <= 320
