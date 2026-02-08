import json

from agents.policy_agent import PolicyAgent


class DummyClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def _last_payload(client: DummyClient):
    assert client.published, "expected at least one publish"
    topic, payload = client.published[0]
    return topic, json.loads(payload)


def test_publish_click_primary_includes_target_norm_and_px():
    agent = PolicyAgent()
    agent.teacher_action = {"text": "click left"}
    client = DummyClient()

    chosen = {
        "label": "click_primary",
        "target_norm": [0.25, 0.75],
        "target_px": [320, 640],
    }
    agent._publish_action(client, chosen, {"label": "mouse_move"})

    topic, payload = _last_payload(client)
    assert payload["action"] == "click_primary"
    assert payload.get("target_norm") == [0.25, 0.75]
    assert payload.get("target_px") == [320, 640]


def test_consecutive_moves_reset_only_on_active_actions():
    agent = PolicyAgent()
    client = DummyClient()

    agent._publish_action(client, {"label": "mouse_move", "dx": 1, "dy": 1}, None)
    assert agent.consecutive_moves == 1

    agent._publish_action(client, {"label": "wait"}, None)
    assert agent.consecutive_moves == 1

    agent._publish_action(client, {"label": "mouse_move", "dx": 1, "dy": 1}, None)
    assert agent.consecutive_moves == 2

    agent._publish_action(client, {"label": "key_press", "key": "q"}, None)
    assert agent.consecutive_moves == 0

    agent._publish_action(client, {"label": "mouse_move", "dx": 1, "dy": 1}, None)
    assert agent.consecutive_moves == 1

    agent._publish_action(client, {"label": "click_primary", "target_norm": [0.5, 0.5]}, None)
    assert agent.consecutive_moves == 0


def test_publish_includes_decision_trace_fields():
    agent = PolicyAgent()
    agent.teacher_action = {"text": "move mouse"}
    client = DummyClient()

    chosen = {
        "label": "click_primary",
        "target_norm": [0.5, 0.5],
        "source": "option_layer",
        "reason": "option_explore",
    }
    policy_action = {
        "label": "wait",
        "source": "model",
        "reason": "final_fallback",
    }
    agent._publish_action(client, chosen, policy_action)

    _topic, payload = _last_payload(client)
    assert payload.get("decision_source") == "option_layer"
    assert payload.get("decision_reason") == "option_explore"
    assert payload.get("policy_source") == "model"
    assert payload.get("policy_reason") == "final_fallback"
