Tactical LLM Adapter
====================

The tactical adapter turns a short summary into a shared strategy update. It
is designed to run asynchronously so the reflex loop stays fast.

Interface
---------

`TacticalLLMAdapter.plan(summary, context=None)` returns a `strategy_update`
dict with:

- `global_strategy` (dict with at least `mode` and `ts`)
- `targets` (dict)
- `cooldowns` (dict)

Backends
--------

- `remote_http` (default): calls a chat-completions HTTP endpoint.
- `trt_llm`: alias of remote HTTP with TRT-specific defaults (for Jetson).

Environment variables
---------------------

- `TACTICAL_LLM_BACKEND` (default: `remote_http`)
- `TACTICAL_LLM_ENDPOINT` (default: `LLM_ENDPOINT` or `http://127.0.0.1:8000/v1/chat/completions`)
- `TACTICAL_LLM_MODEL` (default: `LLM_MODEL` or `llama`)
- `TACTICAL_LLM_TIMEOUT_SEC` (default: `8.0`)
- `TACTICAL_LLM_TEMPERATURE` (default: `0.2`)
- `TACTICAL_LLM_API_KEY` (optional)
- `TACTICAL_DEFAULT_MODE` (default: `scan`)

TRT-LLM overrides:

- `TACTICAL_TRT_LLM_ENDPOINT`
- `TACTICAL_TRT_LLM_MODEL`
- `TACTICAL_TRT_LLM_TIMEOUT_SEC`
- `TACTICAL_TRT_LLM_TEMPERATURE`
- `TACTICAL_TRT_LLM_API_KEY`

Usage
-----

```python
from tactical.llm_adapter import TacticalLLMAdapter

adapter = TacticalLLMAdapter()
update = adapter.plan({"scene": "combat", "objects": ["enemy"]})
```

If the LLM fails or returns invalid JSON, the adapter returns a default update
with `mode=scan` and empty targets/cooldowns.
