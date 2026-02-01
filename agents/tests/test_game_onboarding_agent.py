import json

from agents import game_onboarding_agent as goa


class DummyClient:
    def __init__(self):
        self.calls = []

    def publish(self, topic, payload, retain=False):
        self.calls.append((topic, payload, retain))


def test_schema_publish_retained():
    agent = goa.GameOnboardingAgent()
    agent.client = DummyClient()
    schema = {
        "game_id": "test_game",
        "ui_layout": {},
        "controls": {},
        "signals": {},
        "profile": {},
        "profile_status": "ok",
        "llm_status": "ok",
        "notes": [],
    }
    agent._publish_schema(schema)
    retained = [call for call in agent.client.calls if call[0] == goa.SCHEMA_TOPIC]
    assert retained, "Schema publish missing"
    assert retained[0][2] is True, "Schema publish must be retained"
