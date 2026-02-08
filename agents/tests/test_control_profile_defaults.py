from agents.control_profile import safe_profile


def test_safe_profile_default_rate_limit_is_not_too_low():
    profile = safe_profile()
    assert int(profile.get("max_actions_per_window", 0)) >= 20
    assert float(profile.get("window_sec", 0.0)) <= 10.0
