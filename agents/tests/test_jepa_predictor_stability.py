import importlib
from pathlib import Path

import torch


def _reload_module(monkeypatch):
    import agents.jepa_predictor_agent as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("JEPA_ONLINE_TRAINING", "0")
    return importlib.reload(module)


def test_quarantines_bad_checkpoint_on_load_exception(monkeypatch, tmp_path):
    module = _reload_module(monkeypatch)
    checkpoint_path = tmp_path / "predictor.pt"
    checkpoint_path.write_bytes(b"not-a-valid-checkpoint")
    monkeypatch.setattr(module, "MODEL_PATH", checkpoint_path)

    reasons = []

    def fake_quarantine(self, reason):
        reasons.append(reason)

    monkeypatch.setattr(module.JepaPredictorAgent, "_quarantine_bad_checkpoint", fake_quarantine)

    module.JepaPredictorAgent()

    assert reasons and "load error" in reasons[-1]


def test_non_finite_pred_error_skips_publish_and_recovers(monkeypatch):
    module = _reload_module(monkeypatch)
    published = []
    recovered = []

    class _Client:
        def publish(self, topic, payload):
            published.append((topic, payload))

    agent = module.JepaPredictorAgent()
    agent.client = _Client()

    def fake_recover(reason):
        recovered.append(reason)

    agent._recover_models = fake_recover
    agent.criterion = lambda _pred, _next: torch.tensor(float("nan"))

    z_t = torch.zeros(1, module.EMBED_DIM)
    z_next = torch.zeros(1, module.EMBED_DIM)
    action = {"action": "wait"}
    agent.process_transition(z_t, action, z_next)

    assert recovered and "non-finite pred_error" in recovered[-1]
    assert not published
