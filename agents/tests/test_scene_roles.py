import json
import time

from agents.scene_agent import SceneAgent


class _FakeClient:
    def __init__(self):
        self.messages = []

    def publish(self, _topic, payload):
        self.messages.append(payload)


def _latest_scene_update(messages):
    for raw in reversed(messages):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if payload.get("event") == "scene_update":
            return payload
    return None


def test_extract_roles_splits_world_and_ui_objects():
    agent = SceneAgent()
    objects = [
        {"label": "person", "confidence": 0.9, "bbox": [0.45, 0.45, 0.55, 0.7]},
        {"label": "person", "confidence": 0.6, "bbox": [0.68, 0.45, 0.76, 0.62]},
        {"label": "creature", "confidence": 0.7, "bbox": [0.82, 0.08, 0.96, 0.3]},
        {"label": "traffic light", "confidence": 0.8, "bbox": [0.03, 0.78, 0.19, 0.98]},
        {"label": "chest", "confidence": 0.75, "bbox": [0.58, 0.5, 0.65, 0.61]},
    ]
    player = agent._extract_player(objects)
    enemies = agent._extract_enemies(objects, player)
    roles = agent._extract_roles(objects, player, enemies, resources={})

    stats = roles.get("stats") or {}
    assert stats.get("hostile_count", 0) >= 1
    assert stats.get("interactable_count", 0) >= 1
    assert stats.get("ui_count", 0) >= 1

    for hostile in roles.get("hostiles") or []:
        bbox = hostile.get("bbox") or []
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        assert not (cx >= 0.75 and cy <= 0.35)


def test_scene_update_includes_roles_and_role_stats():
    agent = SceneAgent()
    now = time.time()
    agent.state["snapshot_ts"] = now
    agent.state["mean"].append(0.3)
    agent.state["easy_text"] = "Score: 123"
    agent.state["simple_text"] = ""
    agent.state["flags"] = {"in_game": True}
    agent.state["objects"] = [
        {"label": "person", "confidence": 0.9, "bbox": [0.44, 0.44, 0.56, 0.7]},
        {"label": "person", "confidence": 0.6, "bbox": [0.67, 0.44, 0.75, 0.63]},
    ]
    agent.state["text_zones"] = {"hud": {"text": "Score: 123", "bbox": [0.4, 0.0, 0.6, 0.08]}}
    client = _FakeClient()

    agent._maybe_publish(client)
    payload = _latest_scene_update(client.messages)
    assert payload is not None
    roles = payload.get("roles") or {}
    assert isinstance(roles, dict)
    assert isinstance(roles.get("stats"), dict)
    assert "hostile_count" in roles["stats"]
    assert "resource_count" in roles["stats"]
