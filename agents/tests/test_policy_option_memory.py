import time

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


def _state(now: float):
    return {
        "ok": True,
        "event": "scene_update",
        "timestamp": now,
        "mean": 0.25,
        "flags": {"in_game": True, "death": False},
        "text": ["Quest marker nearby"],
        "objects": [],
    }


def test_option_memory_records_and_replays_success():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    action = {"label": "click_primary", "target_norm": [0.52, 0.31]}

    key = agent._option_memory_key(state)
    agent._update_option_memory(key, action, ok=True)

    replay = agent._memory_option_action(state)
    assert replay is not None
    assert replay.get("label") == "click_primary"
    assert replay.get("target_norm") == [0.52, 0.31]
    assert replay.get("reason") == "episodic_memory"


def test_option_memory_does_not_learn_wait():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    key = agent._option_memory_key(state)

    agent._update_option_memory(key, {"label": "wait"}, ok=True)

    replay = agent._memory_option_action(state)
    assert replay is None


def test_option_layer_produces_non_wait_idle_action():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    agent.current_intent = "idle"

    action = agent._option_layer_action(state)
    assert action is not None
    assert action.get("label") in {"mouse_move", "click_primary"}
    assert action.get("reason") == "option_explore"


def test_option_memory_skips_noop_mouse_move_replay():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    key = agent._option_memory_key(state)
    agent.cursor_x_norm = 0.5
    agent.cursor_y_norm = 0.5

    agent._update_option_memory(
        key,
        {"label": "mouse_move", "target_norm": [0.5, 0.5], "dx": 0, "dy": 0},
        ok=True,
    )

    replay = agent._memory_option_action(state)
    assert replay is None


def test_option_memory_skips_hud_corner_click_replay():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    key = agent._option_memory_key(state)

    agent._update_option_memory(
        key,
        {"label": "click_primary", "target_norm": [0.13, 0.78]},
        ok=True,
    )

    replay = agent._memory_option_action(state)
    assert replay is None


def test_option_memory_caps_repeated_replay():
    agent = PolicyAgent()
    now = time.time()
    state = _state(now)
    key = agent._option_memory_key(state)
    old_max_repeat = policy.POLICY_OPTION_MEMORY_MAX_REPEAT
    old_cooldown = policy.POLICY_OPTION_MEMORY_REPEAT_COOLDOWN_SEC
    try:
        policy.POLICY_OPTION_MEMORY_MAX_REPEAT = 2
        policy.POLICY_OPTION_MEMORY_REPEAT_COOLDOWN_SEC = 60.0
        agent._update_option_memory(
            key,
            {"label": "click_primary", "target_norm": [0.51, 0.47]},
            ok=True,
        )
        assert agent._memory_option_action(state) is not None
        assert agent._memory_option_action(state) is not None
        assert agent._memory_option_action(state) is None
    finally:
        policy.POLICY_OPTION_MEMORY_MAX_REPEAT = old_max_repeat
        policy.POLICY_OPTION_MEMORY_REPEAT_COOLDOWN_SEC = old_cooldown
