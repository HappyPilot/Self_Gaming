import importlib

import torch


def _reload_module(monkeypatch):
    import agents.jepa_predictor_agent as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("JEPA_ONLINE_TRAINING", "0")
    return importlib.reload(module)


def test_pred_error_payload_contains_pred_error(monkeypatch):
    module = _reload_module(monkeypatch)
    published = []

    class _Client:
        def publish(self, topic, payload):
            published.append((topic, payload))

    agent = module.JepaPredictorAgent()
    agent.client = _Client()

    z_t = torch.zeros(1, module.EMBED_DIM)
    z_next = torch.zeros(1, module.EMBED_DIM)
    action = {"action": "wait"}
    agent.process_transition(z_t, action, z_next)

    assert published, "Expected pred_error publish"
    topic, payload = published[-1]
    assert topic == module.PRED_ERROR_TOPIC
    assert "\"pred_error\":" in payload
