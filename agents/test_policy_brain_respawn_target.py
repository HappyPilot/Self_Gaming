import time

from policy_brain import HeuristicPolicyModel


def test_death_click_cooldown_returns_wait():
    model = HeuristicPolicyModel()
    obs = {"flags": {"death": True}}

    first = model(obs)
    second = model(obs)

    assert first["action_type"] in {"click_primary", "click_secondary"}
    assert second["action_type"] == "wait"
