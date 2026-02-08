import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


def _state_with_text(texts, in_game=True, mean=None):
    state = {"flags": {"in_game": in_game}, "text": texts}
    if mean is not None:
        state["mean"] = mean
    return state


def test_scene_allows_action_in_game_bypasses_game_keyword_gate():
    agent = PolicyAgent()
    agent.game_keywords = {"anything"}
    allowed, reason = agent._scene_allows_action(_state_with_text([], in_game=True, mean=None))
    assert allowed is True
    assert reason is None


def test_scene_allows_action_in_game_ignores_desktop_keywords():
    agent = PolicyAgent()
    agent.desktop_keywords = {"finder", "wallpaper"}
    allowed, reason = agent._scene_allows_action(_state_with_text(["Finder", "Wallpaper"], in_game=True))
    assert allowed is True
    assert reason is None


def test_scene_allows_action_uses_game_evidence_when_in_game_flag_false():
    agent = PolicyAgent()
    state = {
        "flags": {"in_game": False, "front_app": "Terminal"},
        "text": ["Resurrect at checkpoint", "Life 0/81", "Mana 68/68"],
        "resources": {
            "life": {"current": 0, "max": 81},
            "mana": {"current": 68, "max": 68},
        },
        "mean": 0.2,
    }
    allowed, reason = agent._scene_allows_action(state)
    assert allowed is True
    assert reason is None


def test_infer_intent_recovers_when_in_game_flag_false_but_respawn_visible():
    agent = PolicyAgent()
    state = {
        "flags": {"in_game": False, "front_app": "Terminal"},
        "text": ["Resurrect at checkpoint", "Life 0/81"],
        "resources": {"life": {"current": 0, "max": 81}},
    }
    intent, reason = agent._infer_intent(state)
    assert intent == "recover"
    assert reason == "respawn_text"


def test_proactive_idle_clicks_near_center():
    agent = PolicyAgent()
    agent.stage0_enabled = False
    agent.current_task = None
    agent.respawn_macro_active = False
    agent.respawn_pending = False
    agent.current_intent = "idle"
    agent.intent_enabled = True
    agent.player_center = [0.5, 0.5]
    policy.POLICY_RANDOM_FALLBACK = True
    policy.PROACTIVE_IDLE_CHANCE = 1.0
    policy.PROACTIVE_IDLE_RADIUS = 0.05
    policy.PROACTIVE_IDLE_MOVE_PROB = 1.0
    # Ensure intent stays idle and model returns no action so proactive idle can trigger.
    agent._update_intent = lambda _state: None
    agent._action_from_model = lambda _state: None
    action = agent._policy_from_observation({"flags": {"in_game": True}})
    assert action is not None
    assert action.get("label") == "mouse_move"
    target = action.get("target_norm")
    assert target is not None
    assert 0.45 <= target[0] <= 0.55
    assert 0.45 <= target[1] <= 0.55


def test_wait_streak_break_forces_exploration_move():
    agent = PolicyAgent()
    agent.current_task = None
    agent.respawn_macro_active = False
    agent.respawn_pending = False
    agent.current_intent = "observe"
    agent.player_center = [0.5, 0.5]
    agent.cursor_x_norm = 0.5
    agent.cursor_y_norm = 0.5
    agent.consecutive_waits = 12

    policy.PROACTIVE_IDLE_MOVE_PROB = 1.0

    action = agent._maybe_break_wait_stall({"label": "wait"}, {"flags": {"in_game": True}})
    assert action is not None
    assert action.get("label") == "mouse_move"
    assert action.get("reason") == "wait_stall_break"


def test_model_wait_falls_back_to_option_layer():
    agent = PolicyAgent()
    agent.current_task = None
    agent.respawn_macro_active = False
    agent.respawn_pending = False
    agent.current_intent = "observe"
    agent.intent_enabled = False
    agent.last_action_ts = 0.0
    agent._action_from_model = lambda _state: {"label": "wait"}
    agent._option_layer_action = lambda _state: {"label": "click_primary", "reason": "option_explore"}

    action = agent._policy_from_observation({"flags": {"in_game": True}, "mean": 0.2})
    assert action is not None
    assert action.get("label") == "click_primary"
    assert action.get("reason") == "option_explore"


def test_proactive_idle_allows_game_evidence_when_in_game_flag_false():
    agent = PolicyAgent()
    old_chance = policy.PROACTIVE_IDLE_CHANCE
    try:
        policy.PROACTIVE_IDLE_CHANCE = 1.0
        state = {
            "flags": {"in_game": False, "front_app": "Terminal"},
            "text": ["Life 10/81", "Mana 30/68"],
            "resources": {
                "life": {"current": 10, "max": 81},
                "mana": {"current": 30, "max": 68},
            },
        }
        action = agent._maybe_proactive_idle_action(state)
    finally:
        policy.PROACTIVE_IDLE_CHANCE = old_chance
    assert action is not None
    assert action.get("label") in {"mouse_move", "click_primary"}
