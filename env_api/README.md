Universal Env API
=================

The Env API standardizes game interaction behind a small, gym-style
interface. It builds on the observation and action schemas introduced in
`schemas/observation.schema.json` and `schemas/action.schema.json`.

Interface
---------

`EnvAdapter` defines:

- `reset()` -> `Observation`
- `step(action)` -> `StepResult`
- `get_observation(timeout_sec=None)` -> `Observation | None`
- `health_check()` -> `dict`
- `close()`

`StepResult.done` is `False` unless the observation payload includes a
`done` (or `terminal`) flag.

MQTT adapter
------------

`MqttEnvAdapter` listens for observations on `vision/observation` (or
`ENV_OBS_TOPIC`) and publishes actions to `act/cmd` (or `ENV_ACTION_TOPIC`).

Example:

```bash
python - <<'PY'
from env_api.mqtt_env import MqttEnvAdapter

env = MqttEnvAdapter()
obs = env.reset()
result = env.step({"action": "mouse_move", "dx": 5, "dy": -2})
print(result.observation)
env.close()
PY
```

Environment variables:

- `ENV_MQTT_CLIENT_ID` (default: env_adapter_<pid>_<suffix>)
- `ENV_OBS_TOPIC` (default: `vision/observation`)
- `ENV_ACTION_TOPIC` (default: `act/cmd`)
- `ENV_REWARD_TOPIC` (default: `train/reward`)
- `ENV_STEP_TIMEOUT_SEC` (default: `1.0`)
- `ENV_OBS_TIMEOUT_SEC` (default: `2.0`)
- `ENV_HEALTH_STALE_SEC` (default: `5.0`)
- `ENV_SCHEMA_VALIDATE` (default: `0`, enable payload validation)
- `ENV_REWARD_SKEW_SEC` (default: `0.2`, reward timestamp tolerance)

Replay integration
------------------

Use the replay harness to feed observations back into MQTT, then consume
them via `MqttEnvAdapter`:

```bash
python -m replay.replay_runner /mnt/ssd/datasets/<game>/<session_id>
```

Windows play client
-------------------

The Windows-side play client should implement the same `EnvAdapter` interface
over capture + input injection. Use `env_api/adapter.py` as the contract.

Testing helper
--------------

`InMemoryEnvAdapter` (in `env_api/in_memory.py`) can be used to unit-test
policies without MQTT.
