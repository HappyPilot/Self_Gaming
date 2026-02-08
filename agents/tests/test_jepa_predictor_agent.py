import torch

from agents.jepa_predictor_agent import EMBED_DIM, JepaPredictorAgent


def test_handle_embedding_uses_predictor_dtype():
    agent = JepaPredictorAgent()
    agent.predictor.half()
    expected_dtype = next(agent.predictor.parameters()).dtype
    seen = {}

    def _capture_transition(_z_t, _action, z_next):
        seen["dtype"] = z_next.dtype

    agent.process_transition = _capture_transition  # type: ignore[method-assign]
    agent.last_z = torch.zeros((1, EMBED_DIM), dtype=expected_dtype, device=next(agent.predictor.parameters()).device)
    agent.last_action = {"action": "wait"}
    agent.handle_embedding({"embedding": [0.0] * EMBED_DIM})

    assert seen.get("dtype") == expected_dtype
