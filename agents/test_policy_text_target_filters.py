from agents.policy_agent import PolicyAgent


def test_pick_text_target_ignores_caps_long_labels():
    agent = PolicyAgent()
    targets = [
        {"label": "QUEST OBJECTIVE", "center": [0.2, 0.2]},
        {"label": "Chest", "center": [0.6, 0.6]},
    ]
    picked = agent._pick_text_target(targets)
    assert picked is not None
    center, label = picked
    assert label == "Chest"
    assert center == [0.6, 0.6]


def test_update_scene_targets_filters_non_actionable_quest_header():
    agent = PolicyAgent()
    state = {
        "targets": [
            {"label": "ENEMYATTHEGATE", "center": [0.87, 0.31], "bbox": [0.82, 0.29, 0.91, 0.33]},
            {"label": "Large Chest", "center": [0.55, 0.44], "bbox": [0.50, 0.40, 0.60, 0.48]},
        ]
    }

    agent._update_scene_targets(state)

    labels = [str(item.get("label")) for item in agent.scene_targets]
    assert "ENEMYATTHEGATE" not in labels
    assert "Large Chest" in labels


def test_attach_teacher_target_hint_drops_non_actionable_ui_target():
    agent = PolicyAgent()
    payload = {
        "text": "click left target:(0.87,0.31)",
        "reasoning": "target_hint: ENEMYATTHEGATE",
    }

    agent._attach_teacher_target_hint(payload)

    assert "target_norm" not in payload


def test_attach_teacher_target_hint_drops_top_right_ui_coordinate_without_label():
    agent = PolicyAgent()
    payload = {
        "text": "move mouse (0.87,0.31)",
        "reasoning": "scan screen",
    }

    agent._attach_teacher_target_hint(payload)

    assert "target_norm" not in payload
