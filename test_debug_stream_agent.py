from PIL import Image

from agents import debug_stream_agent as ds


def test_control_point_from_action_target_norm():
    action = {"target_norm": [0.25, 0.5]}
    assert ds.control_point_from_action(action, 200, 100) == (50, 50)


def test_control_point_from_action_dx_dy():
    action = {"dx": 10, "dy": -5}
    assert ds.control_point_from_action(action, 100, 100) == (60, 45)


def test_build_debug_payload():
    img = Image.new("RGB", (10, 20), color=(0, 0, 0))
    payload = ds.build_debug_payload(img, 123.0)
    assert payload["ok"] is True
    assert payload["width"] == 10
    assert payload["height"] == 20
    assert payload["timestamp"] == 123.0
    assert payload["source"] == "debug_stream"
    assert "image_b64" in payload


def test_should_publish_debug():
    assert ds.should_publish_debug(100.0, 100.1, fps=1.0) is False
    assert ds.should_publish_debug(100.0, 101.1, fps=1.0) is True
