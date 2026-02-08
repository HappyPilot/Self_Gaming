import json
import time

import agents.scene_agent as scene
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


def _base_agent(in_game: bool) -> SceneAgent:
    agent = SceneAgent()
    now = time.time()
    agent.state["snapshot_ts"] = now
    agent.state["mean"].append(0.2)
    agent.state["easy_text"] = ""
    agent.state["simple_text"] = ""
    agent.state["objects"] = []
    agent.state["text_zones"] = {}
    agent.state["flags"] = {"in_game": in_game}
    agent.state["flags_ts"] = now
    agent.state["prompt_scores"] = {"quest": 0.95}
    agent.state["prompt_ts"] = now
    return agent


def test_prompt_tokens_hidden_when_not_in_game():
    agent = _base_agent(in_game=False)
    cli = _FakeClient()
    old_min = scene.PROMPT_TAG_MIN_SCORE
    old_topk = scene.PROMPT_TAG_TOP_K
    old_allow = set(scene.PROMPT_TAG_ALLOW)
    scene.PROMPT_TAG_MIN_SCORE = 0.01
    scene.PROMPT_TAG_TOP_K = 3
    scene.PROMPT_TAG_ALLOW = {"quest"}

    try:
        agent._maybe_publish(cli)
    finally:
        scene.PROMPT_TAG_MIN_SCORE = old_min
        scene.PROMPT_TAG_TOP_K = old_topk
        scene.PROMPT_TAG_ALLOW = old_allow

    assert cli.messages, "scene agent did not publish payload"
    payload = _latest_scene_update(cli.messages)
    assert payload is not None, "scene_update payload not found"
    text = payload.get("text") or []
    assert all(not str(item).startswith("prompt_") for item in text)


def test_prompt_tokens_present_when_in_game():
    agent = _base_agent(in_game=True)
    cli = _FakeClient()
    old_min = scene.PROMPT_TAG_MIN_SCORE
    old_topk = scene.PROMPT_TAG_TOP_K
    old_allow = set(scene.PROMPT_TAG_ALLOW)
    scene.PROMPT_TAG_MIN_SCORE = 0.01
    scene.PROMPT_TAG_TOP_K = 3
    scene.PROMPT_TAG_ALLOW = {"quest"}

    try:
        agent._maybe_publish(cli)
    finally:
        scene.PROMPT_TAG_MIN_SCORE = old_min
        scene.PROMPT_TAG_TOP_K = old_topk
        scene.PROMPT_TAG_ALLOW = old_allow

    assert cli.messages, "scene agent did not publish payload"
    payload = _latest_scene_update(cli.messages)
    assert payload is not None, "scene_update payload not found"
    text = payload.get("text") or []
    assert any(str(item).startswith("prompt_") for item in text)
