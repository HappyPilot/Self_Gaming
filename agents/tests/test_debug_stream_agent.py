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


def test_render_annotated_frame_draws_xyxy_box():
    frame = Image.new("RGB", (100, 100), color=(0, 0, 0))
    objects = [{"box": [10, 20, 30, 40], "class": "enemy"}]

    out = ds.render_annotated_frame(frame, objects, None, "", None)

    # top-left corner should be green outline
    assert out.getpixel((10, 20)) == (0, 255, 0)
    # bottom-right corner of xyxy should also be outlined
    assert out.getpixel((30, 40)) == (0, 255, 0)


def test_render_annotated_frame_draws_normalized_xyxy_bbox():
    frame = Image.new("RGB", (100, 100), color=(0, 0, 0))
    objects = [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "enemy"}]

    out = ds.render_annotated_frame(frame, objects, None, "", None)

    assert out.getpixel((10, 20)) == (0, 255, 0)
    assert out.getpixel((30, 40)) == (0, 255, 0)


def test_on_scene_message_updates_scene_text_and_objects():
    class Msg:
        topic = ds.SCENE_TOPIC
        payload = b'{"text":["enemy nearby"],"objects":[{"bbox":[0.1,0.2,0.3,0.4],"label":"enemy"}]}'

    ds.latest_scene_text = ""
    ds.latest_objects = []
    ds.on_message(None, None, Msg())

    assert ds.latest_scene_text == "enemy nearby"
    assert len(ds.latest_objects) == 1


def test_on_scene_message_builds_overlay_from_player_and_roles_when_objects_empty():
    class Msg:
        topic = ds.SCENE_TOPIC
        payload = (
            b'{"text":["alive"],'
            b'"objects":[],'
            b'"player":{"bbox":[0.4,0.4,0.6,0.7],"label":"hero"},'
            b'"roles":{"interactables":[{"bbox":[0.7,0.5,0.8,0.6],"label":"chest"}]}}'
        )

    ds.latest_scene_text = ""
    ds.latest_objects = []
    ds.on_message(None, None, Msg())

    assert ds.latest_scene_text == "alive"
    assert len(ds.latest_objects) >= 2
