import importlib

import torch


def test_action_encoder_matches_projection_dtype(monkeypatch):
    import agents.jepa_predictor_agent as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module = importlib.reload(module)

    encoder = module.ActionEncoder(out_dim=module.ACTION_DIM).half()
    out = encoder({"action": "mouse_move", "dx": 4, "dy": -2, "target_norm": [0.5, 0.5]})
    assert out.dtype == torch.float16
