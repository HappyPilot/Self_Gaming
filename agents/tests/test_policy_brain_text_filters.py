import agents.policy_brain as policy_brain


def test_select_target_skips_ui_text():
    targets = [
        {"label": "ENEMY AT THE GATE", "center": [0.2, 0.2]},
        {"label": "Chest", "center": [0.6, 0.6]},
    ]
    target = policy_brain._select_target(targets)
    assert target is not None
    assert target["label"] == "Chest"


def test_select_target_skips_caps_long_text():
    targets = [
        {"label": "QUEST OBJECTIVE", "center": [0.2, 0.2]},
    ]
    assert policy_brain._select_target(targets) is None


def test_text_target_probability_can_disable_text_targets():
    model = policy_brain.HeuristicPolicyModel()
    original_prob = policy_brain.TEXT_TARGET_PROB
    try:
        policy_brain.TEXT_TARGET_PROB = 0.0
        scene = {
            "flags": {"in_game": True},
            "targets": [{"label": "Chest", "center": [0.6, 0.6]}],
        }
        action = model(scene, vec=None)
        assert action["action_type"] == policy_brain.IDLE_ACTION
    finally:
        policy_brain.TEXT_TARGET_PROB = original_prob


def test_ignore_tokens_do_not_include_specific_phrases():
    assert "atthegate" not in policy_brain.IGNORE_TEXT_TOKENS
    assert "enemyatthegate" not in policy_brain.IGNORE_TEXT_TOKENS
