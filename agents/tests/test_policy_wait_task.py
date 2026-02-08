import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


def test_wait_task_respects_duration(monkeypatch):
    agent = PolicyAgent()
    task = {"action_type": "WAIT", "duration": 1.5, "task_id": "wait_1"}
    agent.current_task = task

    now = [1000.0]
    monkeypatch.setattr(policy.time, "time", lambda: now[0])

    action = agent._action_from_task({})
    assert action["label"] == "wait"
    assert action.get("task_complete") is False
    agent._maybe_advance_task(action)
    assert agent.current_task == task

    now[0] += 2.0
    action = agent._action_from_task({})
    assert action["label"] == "wait"
    assert action.get("task_complete") is True
    agent._maybe_advance_task(action)
    assert agent.current_task is None
