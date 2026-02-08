from pathlib import Path


def _compose_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "docker-compose.yml").read_text(encoding="utf-8")


def test_jepa_predictor_uses_configurable_cuda_visibility():
    data = _compose_text()
    marker = "jepa_predictor:"
    i = data.find(marker)
    assert i != -1, "jepa_predictor service not found"
    end = data.find("\n  siglip_prompt:", i + len(marker))
    block = data[i : end if end != -1 else len(data)]
    assert (
        "CUDA_VISIBLE_DEVICES=${JEPA_CUDA_VISIBLE_DEVICES:-0}" in block
    ), "jepa_predictor should default to GPU and allow override via JEPA_CUDA_VISIBLE_DEVICES"


def test_zeroshot_detector_forces_cpu_visibility():
    data = _compose_text()
    marker = "zeroshot_detector:"
    i = data.find(marker)
    assert i != -1, "zeroshot_detector service not found"
    block = data[i : data.find("\n  scene:", i + len(marker)) if data.find("\n  scene:", i + len(marker)) != -1 else len(data)]
    assert "CUDA_VISIBLE_DEVICES=" in block, "zeroshot_detector should hide CUDA devices to avoid starving vl_jepa"
