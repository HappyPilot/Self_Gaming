from pathlib import Path


def _compose_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "docker-compose.yml").read_text(encoding="utf-8")


def _service_block(data: str, service: str, next_service: str) -> str:
    marker = f"  {service}:"
    i = data.find(marker)
    assert i != -1, f"{service} service not found"
    end_marker = f"\n  {next_service}:"
    j = data.find(end_marker, i + len(marker))
    return data[i : j if j != -1 else len(data)]


def test_perception_service_does_not_force_cpu_device():
    data = _compose_text()
    block = _service_block(data, "perception", "hint_client")
    assert "YOLO11_DEVICE=cpu" not in block
    assert "YOLO11_DEVICE=${YOLO11_DEVICE:-cuda:0}" in block


def test_scene_observation_topic_is_configurable():
    data = _compose_text()
    block = _service_block(data, "scene", "embedding_guard")
    assert "VISION_OBSERVATION_TOPIC=${VISION_OBSERVATION_TOPIC:-vision/observation}" in block
