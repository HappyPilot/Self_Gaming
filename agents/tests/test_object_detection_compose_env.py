from pathlib import Path


def _compose_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "docker-compose.yml").read_text(encoding="utf-8")


def test_object_detection_pins_low_mode_and_queue():
    data = _compose_text()
    marker = "object_detection:"
    i = data.find(marker)
    assert i != -1, "object_detection service not found"
    end = data.find("\n  vl_jepa:", i + len(marker))
    block = data[i : end if end != -1 else len(data)]
    assert "VISION_MODE_DEFAULT=low" in block
    assert "VISION_CONFIG_TOPIC=" in block
    assert "OBJECT_QUEUE=8" in block
