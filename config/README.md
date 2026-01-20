# Config Profiles

This repo uses env profiles for Docker Compose.

- Defaults: `config/defaults.env`
- Profiles: `config/jetson.env`, `config/mac.env`, `config/production.env`

## Select a profile
```bash
export SG_PROFILE=jetson
```

Examples:
```bash
export SG_PROFILE=mac
export SG_PROFILE=production
```
Notes:
- `mac` profile uses `OCR_BACKEND=easyocr` and `OCR_FORCE_CPU=1`.
- `production` keeps defaults and expects explicit overrides.

## Local overrides (not committed)
If you need host-specific endpoints, copy `config/local.env.example` to
`config/local.env` and pass it via `SG_LOCAL_ENV_FILE` (use an absolute path):
```bash
export SG_LOCAL_ENV_FILE="$(pwd)/config/local.env"
```

## Control profiles (auto game_id)
Set `CONTROL_PROFILE_GAME_ID=auto` to let training use the last onboarded `game_schema` game_id.
Use `ONBOARD_GAME_ID_OVERRIDE=path_of_exile` to force a specific game during onboarding (optional).

## Game identity (window/process)
On the Mac host, run `tools/mac/game_identity_publisher.py` to publish the frontmost app/window to MQTT.
Jetson listens on `GAME_IDENTITY_TOPIC=game/identity` and stores `game_identity` in memory for onboarding/training.

## LLM endpoint overrides
Point Jetson agents at a remote LLM server by overriding these:
```bash
LLM_ENDPOINT=http://10.0.0.230:11434/v1/chat/completions
TEACHER_LOCAL_ENDPOINT=http://10.0.0.230:11434/v1/chat/completions
```
`LLM_MODEL` must match a model ID returned by `GET /v1/models` on that endpoint. If you are unsure, set `LLM_MODEL=auto` and the onboarding client will pick the first available model.

## LLM gating (prevent concurrent requests)
To avoid multiple agents competing for the LLM, a shared gate file can be used:
```bash
LLM_GATE_FILE=/mnt/ssd/logs/llm_gate.lock
LLM_GATE_WAIT_S=60
LLM_GATE_TTL_S=120
```
When the onboarding agent runs, it acquires the gate so other agents skip LLM calls.
To pause LLM calls across agents manually:
```bash
touch /mnt/ssd/logs/llm_pause
rm /mnt/ssd/logs/llm_pause
```

## LLM queue service (serialized requests)
Run the local queue and point agents at it:
```bash
LLM_ENDPOINT=http://127.0.0.1:9010/v1/chat/completions
TEACHER_LOCAL_ENDPOINT=http://127.0.0.1:9010/v1/chat/completions
LLM_QUEUE_UPSTREAM=http://10.0.0.230:11434/v1/chat/completions
```
Start the queue service with Docker Compose (`llm_queue`).

## OCR language defaults
`OCR_LANGS` defaults to `en` to reduce OCR latency and false positives.
Enable more languages by overriding `OCR_LANGS` in your local env file.

## Policy targeting overrides
Use these when the policy gets stuck on generic mouse_move actions:
```bash
POLICY_PREFER_TARGETS=1
TEACHER_TARGET_PRIORITY=1
RESPAWN_TRIGGER_TEXTS=respawn,revive,resurrect,resurrect at checkpoint
```
`POLICY_PREFER_TARGETS` lets OCR/object targets override a pure mouse_move.
`TEACHER_TARGET_PRIORITY` makes teacher actions with explicit coordinates take priority over mouse_move.

## Demonstrator key actions
The demonstrator defaults to clicks; allow key presses only if whitelisted:
```bash
DEMO_ALLOW_KEYS=0
```
