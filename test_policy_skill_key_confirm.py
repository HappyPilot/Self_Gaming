import importlib


def _reload_policy(monkeypatch, env):
    import agents.policy_agent as module
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(module)


def test_unconfirmed_extended_key_blocked(monkeypatch):
    module = _reload_policy(
        monkeypatch,
        {
            "POLICY_SKILL_KEYS": "q",
            "POLICY_FALLBACK_SKILL_KEYS": "q",
            "TEACHER_KEY_CONFIRM_HITS": "1",
        },
    )

    agent = module.PolicyAgent()
    agent.profile_allowed_keys = set()
    agent.skill_stats = {}

    assert agent._choose_skill_key() is None


def test_choose_skill_key_uses_tie_break(monkeypatch):
    module = _reload_policy(
        monkeypatch,
        {
            "POLICY_SKILL_KEYS": "1,2",
            "POLICY_FALLBACK_SKILL_KEYS": "",
            "POLICY_SKILL_EPS": "0",
        },
    )

    agent = module.PolicyAgent()
    agent.profile_allowed_keys = set()
    agent.skill_stats = {}
    agent.rng.random = lambda: 0.9
    agent.rng.choice = lambda items: items[-1]

    assert agent._choose_skill_key() == "2"
