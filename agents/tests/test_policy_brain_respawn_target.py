import agents.policy_brain as policy_brain


def test_brain_targets_respawn_on_death_with_cooldown(monkeypatch):
    model = policy_brain.HeuristicPolicyModel()
    scene = {"flags": {"in_game": True, "death": True}}

    now = [1000.0]
    monkeypatch.setattr(policy_brain.time, "time", lambda: now[0])

    action = model(scene, vec=None)
    assert action["action_type"] == "click_primary"
    target = action.get("target")
    assert isinstance(target, dict)
    assert "x" in target and "y" in target

    now[0] += 1.0
    action = model(scene, vec=None)
    assert action["action_type"] == "wait"

    now[0] += 1.1
    action = model(scene, vec=None)
    assert action["action_type"] == "click_primary"
