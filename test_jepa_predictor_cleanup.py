import importlib

import torch


def _reload_module(monkeypatch, env):
    import agents.jepa_predictor_agent as module
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return importlib.reload(module)


def test_startup_jitter_sleeps(monkeypatch):
    module = _reload_module(monkeypatch, {"JEPA_STARTUP_JITTER_SEC": "5"})

    sleep_calls = []

    def fake_sleep(value):
        sleep_calls.append(value)

    monkeypatch.setattr(module.random, "uniform", lambda _a, _b: 3.2)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    module.JepaPredictorAgent()

    assert sleep_calls == [3.2]


def test_cleanup_calls_gc_on_train_step(monkeypatch):
    module = _reload_module(monkeypatch, {"JEPA_CUDA_CLEANUP_EVERY": "1"})

    gc_calls = []
    empty_cache_calls = []

    monkeypatch.setattr(module.gc, "collect", lambda: gc_calls.append(True))
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    agent = module.JepaPredictorAgent()

    z_t = torch.zeros(1, module.EMBED_DIM)
    z_next = torch.zeros(1, module.EMBED_DIM)
    action = {"action": "wait"}

    module.ONLINE_TRAINING = True
    agent.process_transition(z_t, action, z_next)

    assert gc_calls, "gc.collect should be called on cleanup"
    assert not empty_cache_calls, "empty_cache should be skipped on CPU"
