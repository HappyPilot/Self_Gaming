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
