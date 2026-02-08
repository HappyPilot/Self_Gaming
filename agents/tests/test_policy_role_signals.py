import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent, encode_non_visual


def test_encode_non_visual_uses_role_stats_for_counts():
    state = {
        "roles": {
            "stats": {
                "hostile_count": 3,
                "interactable_count": 2,
            }
        },
        "objects": [],
        "resources": {},
    }
    vector = encode_non_visual(state)
    assert float(vector[2].item()) == 3.0
    assert float(vector[3].item()) == 2.0


def test_enemies_present_uses_roles_when_enemy_list_empty():
    agent = PolicyAgent()
    agent.latest_state = {"enemies": [], "roles": {"hostiles": [{"bbox": [0.6, 0.4, 0.7, 0.6], "confidence": 0.5}]}}
    assert agent._enemies_present() is True


def test_enemies_present_in_state_uses_roles_stats():
    state = {"enemies": [], "roles": {"stats": {"hostile_count": 2}}}
    assert PolicyAgent._enemies_present_in_state(state) is True


def test_meaningful_fallback_ignores_ui_dominant_objects():
    agent = PolicyAgent()
    state = {
        "objects": [{"label": "traffic light", "confidence": 0.95, "bbox": [0.02, 0.78, 0.18, 0.98]}],
        "roles": {
            "stats": {"ui_count": 1, "world_count": 0, "hostile_count": 0, "interactable_count": 0},
            "world_objects": [],
            "interactables": [],
            "hostiles": [],
        },
    }
    old_require = policy.POLICY_REQUIRE_EMBEDDINGS
    old_fallback = policy.POLICY_MEANINGFUL_FALLBACK
    old_ui_layout = policy.POLICY_USE_UI_LAYOUT
    policy.POLICY_REQUIRE_EMBEDDINGS = False
    policy.POLICY_MEANINGFUL_FALLBACK = True
    policy.POLICY_USE_UI_LAYOUT = False
    try:
        action = agent._meaningful_fallback_action(state)
    finally:
        policy.POLICY_REQUIRE_EMBEDDINGS = old_require
        policy.POLICY_MEANINGFUL_FALLBACK = old_fallback
        policy.POLICY_USE_UI_LAYOUT = old_ui_layout
    assert action is None


def test_meaningful_fallback_uses_role_world_objects():
    agent = PolicyAgent()
    state = {
        "objects": [],
        "roles": {
            "stats": {"ui_count": 1, "world_count": 1, "hostile_count": 0, "interactable_count": 0},
            "world_objects": [{"label": "world_object", "confidence": 0.8, "bbox": [0.62, 0.42, 0.72, 0.58]}],
            "interactables": [],
            "hostiles": [],
        },
    }
    old_require = policy.POLICY_REQUIRE_EMBEDDINGS
    old_fallback = policy.POLICY_MEANINGFUL_FALLBACK
    old_ui_layout = policy.POLICY_USE_UI_LAYOUT
    policy.POLICY_REQUIRE_EMBEDDINGS = False
    policy.POLICY_MEANINGFUL_FALLBACK = True
    policy.POLICY_USE_UI_LAYOUT = False
    try:
        action = agent._meaningful_fallback_action(state)
    finally:
        policy.POLICY_REQUIRE_EMBEDDINGS = old_require
        policy.POLICY_MEANINGFUL_FALLBACK = old_fallback
        policy.POLICY_USE_UI_LAYOUT = old_ui_layout
    assert action is not None
    assert action.get("label") in {"mouse_move", "click_primary"}
