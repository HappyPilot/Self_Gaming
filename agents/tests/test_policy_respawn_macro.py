import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


def test_respawn_macro_triggers_on_death_flag_without_text(monkeypatch):
    agent = PolicyAgent()
    agent.game_keywords = set()
    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])
    state = {
        "ok": True,
        "event": "scene_update",
        "timestamp": now[0],
        "mean": 0.2,
        "flags": {"in_game": True, "death": True},
        "text": [],
    }
    action = agent._policy_from_observation(state)
    assert agent.respawn_macro_active is True
    assert action is not None


def test_respawn_macro_click_delay(monkeypatch):
    agent = PolicyAgent()
    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])
    agent._start_respawn_macro()
    first_click_time = None
    for when, template in agent.respawn_macro_queue:
        if template.get("label") == "click_primary":
            first_click_time = when
            break
    assert first_click_time is not None
    assert first_click_time - now[0] >= policy.RESPAWN_MACRO_PRE_CLICK_DELAY


def test_respawn_macro_uses_dynamic_scene_target(monkeypatch):
    agent = PolicyAgent()
    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])
    state = {
        "ok": True,
        "event": "scene_update",
        "timestamp": now[0],
        "mean": 0.2,
        "flags": {"in_game": True, "death": True},
        "targets": [
            {"label": "RESURRECT AT CHECKPOINT", "center": [0.51, 0.30]},
        ],
        "text": [],
    }
    action = agent._policy_from_observation(state)
    assert action is not None
    assert action.get("target_norm") == [0.51, 0.30]


def test_respawn_macro_clicks_include_target_norm(monkeypatch):
    agent = PolicyAgent()
    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])
    agent._start_respawn_macro([0.51, 0.30])
    click_templates = [tpl for _when, tpl in agent.respawn_macro_queue if tpl.get("label") == "click_primary"]
    assert click_templates
    assert all(tpl.get("target_norm") == [0.51, 0.30] for tpl in click_templates)


def test_respawn_macro_reuses_last_target_when_new_target_missing(monkeypatch):
    agent = PolicyAgent()
    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])
    agent.respawn_target_norm = [0.51, 0.30]
    agent._start_respawn_macro(None)
    click_templates = [tpl for _when, tpl in agent.respawn_macro_queue if tpl.get("label") == "click_primary"]
    assert click_templates
    assert all(tpl.get("target_norm") == [0.51, 0.30] for tpl in click_templates)
