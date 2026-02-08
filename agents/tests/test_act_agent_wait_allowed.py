from agents.act_agent import ActAgent


def test_wait_action_is_allowed():
    agent = ActAgent()
    allowed, reason = agent._is_allowed({"action": "wait"})
    assert allowed is True
    assert reason == "ok"
