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
On the Mac host, run `tools/mac/game_identity_publisher.py` (or enable `INPUT_PUBLISH_IDENTITY=1` in the input bridge)
to publish the frontmost app/window to MQTT. Jetson listens on `GAME_IDENTITY_TOPIC=game/identity` and stores
`game_identity` in memory for onboarding/training.

To gate actions when the front app is not the game:
```bash
GAME_IDENTITY_PUBLISH_FLAGS=1
GAME_IDENTITY_BIND_MODE=auto
GAME_IDENTITY_GRACE_SEC=2.0
```
`GAME_IDENTITY_BIND_MODE=auto` binds to the first non-unknown app and marks `scene/flags.in_game` true only for that app.

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

To auto-click when the teacher only provides a target (no explicit click intent):
```bash
TEACHER_TARGET_AUTOCLICK=1
TEACHER_TARGET_AUTOCLICK_COOLDOWN=2.0
```

To add a periodic key probe burst during exploration (uses allowed keys from the profile):
```bash
POLICY_EXPLORATION_KEY_BURST=1
POLICY_EXPLORATION_KEY_BURST_COUNT=2
POLICY_EXPLORATION_KEY_BURST_COOLDOWN_SEC=300
```

To require a positive `scene/flags.in_game` signal before any actions:
```bash
POLICY_REQUIRE_IN_GAME=1
POLICY_REQUIRE_IN_GAME_STRICT=1
TEACHER_REQUIRE_IN_GAME=1
TEACHER_REQUIRE_IN_GAME_STRICT=1
DEMO_REQUIRE_IN_GAME=1
DEMO_REQUIRE_IN_GAME_STRICT=1
```

## Exploration keys (safe keyboard probing)
To let exploration use whitelisted keys from the onboarding profile:
```bash
POLICY_EXPLORATION_KEYS_FROM_PROFILE=1
```
This pulls `allowed_keys` from `game/schema` and keeps key presses constrained to known-safe inputs.

## Demonstrator key actions
The demonstrator defaults to clicks; allow key presses only if whitelisted:
```bash
DEMO_ALLOW_KEYS=0
```
