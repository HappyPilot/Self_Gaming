# Multi-Agent UI Automation Stack

This repository hosts the collection of MQTT-driven agents that coordinate UI automation, perception, and learning on the Jetson edge device. Recent additions introduce a language-model-based teacher, YOLO-based object perception, and a dense reward stream wired to the Path of Exile telemetry outlined in `~/Documents/poe_reward_system_v1.md`.

## OpenAI Teacher Agent

Set an OpenAI API key before starting the stack so the teacher agent can call the Chat Completions API:

```bash
export OPENAI_API_KEY="sk-..."
```

The new `teacher_agent` subscribes to `scene/state`, `vision/snapshot`, and `act/result`, asks the LLM for high-level guidance, and publishes actions on `teacher/action`. Responses include both the natural language reasoning and the chosen action string.

### Using a Local LLM Instead of OpenAI

If you prefer not to rely on OpenAI, run a local inference server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint (for example, via `vllm`, `text-generation-webui`, or `ollama serve`). Then set:

```bash
export TEACHER_PROVIDER=local
export TEACHER_LOCAL_ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
```

The teacher agent will POST the same JSON payload it would send to OpenAI, so any service that mimics that schema will work. The `requests` Python package must be available inside the container/virtualenv.

## Policy Blending

`policy_agent.py` now blends its heuristic actions with the teacher’s recommendations. A linear annealing coefficient `teacher_alpha` (configurable via `TEACHER_ALPHA_START` and `TEACHER_ALPHA_DECAY_STEPS`) starts at 1.0, prioritising teacher advice, then decays to rely on the learned policy. The final action is emitted on both `control/keys` and `act/cmd`, preserving the previous behaviour.

## Object-Detection Agent

- `object_detection_agent.py` subscribes to `vision/frame`, runs a pluggable YOLO backend (Ultralytics by default), and emits structured detections on `vision/objects`.
- The agent is packaged in its own container (`local/object-detection-agent`). Drop TensorRT/Ultralytics weights under `/mnt/ssd/models/yolo/<checkpoint>` and point `OBJECT_MODEL_PATH` there.
- Relevant environment switches (see `docker-compose.yml`):
  - `VISION_FRAME_INTERVAL` / `VISION_FRAME_JPEG_QUALITY` – sampling rate + encoding coming from `vision_agent`.
  - `OBJECT_DETECTOR_BACKEND` – `ultralytics` for real inference or `dummy` for smoke tests.
  - `OBJECT_CONF_THRESHOLD`, `OBJECT_IOU_THRESHOLD`, `OBJECT_QUEUE` – runtime tuning knobs.

The `scene_agent` now fuses OCR, mean luminance, and the most recent detection payload so downstream agents receive `objects` with `(class, confidence, box)` triples in every `scene/state` update.

## Reward Manager

`reward_manager.py` pulls three signals to publish dense rewards on `train/reward`:

1. Tail the local PoE client log at `/mnt/ssd/poe_client.txt` (streamed from Windows via the Flask receiver) and normalise entries such as “You have entered…”, “You have killed…”, and “Picked up …”.
2. Consume `scene/state` so the latest YOLO detections contribute enemy density + loot pressure (important for map-progress/loot weights).
3. Watch `act/result` to stay in lockstep with the control loop.

The calculator mirrors the curriculum in `poe_reward_system_v1.md`: stage-specific weights (`S0…S4`), step-costs, death penalties, and loot-value buckets (currency, maps, contracts, etc.). Rewards are clipped to `[-1, 1]` and include a `components` breakdown plus the last few parsed events for debugging.

## Training Pipeline Updates

- `recorder_agent.py` records scene text, YOLO objects, teacher suggestions, and rewards for every `(state, action)` pair in `/mnt/ssd/datasets/episodes`.
- `teach_agent.py` includes the distillation schedule plus the reward topic when it emits `train/jobs`.
- `train_manager_agent.py` consumes the richer samples, applies a KL distillation loss against the teacher actions, and adds a reward-weighted behaviour-cloning/RL loss (tunable via `TEACHER_KL_WEIGHT`, `REWARD_WEIGHT`).

### Workflow

1. Ensure all agents are running (including `teacher_agent` and `reward_manager`).
2. During early training epochs, the policy will follow the teacher’s commands; as `teacher_alpha` decays, it transitions to autonomous behaviour.
3. Rewards emitted on `train/reward` guide the trainer and are stored with each recorded sample.

## Testing

Unit tests cover the teacher prompt pipeline, the policy annealing logic, the object-detection helpers, and the reward calculator:

```bash
python3 -m unittest \
  tests/test_teacher_agent.py \
  tests/test_policy_agent.py \
  tests/test_object_detection_agent.py \
  tests/test_reward_manager.py
```

## Centralised Logging & Log Monitor Agent

- A global `sitecustomize.py` initialises rotating log files for *every* agent. By default logs are written to the first writable path among `LOG_DIR`, `<repo>/logs`, `~/agent_logs`, `/mnt/ssd/logs`, and `/tmp/agent_logs`. Each process receives `<agent_name>.log` plus the usual stdout stream, so Mac-side runs automatically drop files into `self-gaming/logs/` unless you override `LOG_DIR`.
- Tune log behaviour via the following environment variables:
  - `LOG_DIR` – preferred directory (falls back automatically if unavailable).
  - `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` – rotation settings (default 5 MB × 5 files).
  - `LOG_LEVEL` – default `INFO`; set to `DEBUG` for chatty traces.

- To copy Jetson logs onto this Mac, run `tools/sync_jetson_logs.sh` (uses `rsync`).  Override `JETSON_HOST`, `JETSON_LOG_DIR`, or `JETSON_LOGS_DEST` if the defaults (`dima@10.0.0.68:/mnt/ssd/logs` → `logs_jetson/`) need to change.

### Log Monitor Agent

`log_monitor_agent.py` tails every `*.log` file and publishes:

| Topic | Payload |
| --- | --- |
| `logs/summary` | Aggregate counts per agent (`lines`, `warnings`, `errors`). |
| `logs/alerts` | Structured alerts when errors, tracebacks, failed jobs, or NaN losses are detected (rate-limited by `LOG_ALERT_COOLDOWN`, default 30 s). |

Configuration knobs:

- `LOG_MONITOR_DIRS` (colon-separated) or `LOG_MONITOR_DIR` – directories to scan (default `/app/logs`).
- `LOG_MONITOR_INTERVAL` – scan cadence in seconds.
- `LOG_MONITOR_TRAINERS` – comma-separated list of log sources that should trigger the extra training-health heuristics (`train_manager,policy_agent,teach_agent` by default).

Add `log_monitor` to the `docker-compose.yml` stack (already wired) and check `logs/alerts` for actionable messages when training stalls or errors recur.

## MQTT Topics

| Topic | Producer | Purpose |
| --- | --- | --- |
| `teacher/action` | Teacher agent | Natural-language action suggestions + reasoning |
| `train/reward` | Reward manager | Scalar reward events consumed by the trainer |
| `act/cmd` / `control/keys` | Policy agent | Final blended control command |
| `train/status` | Train manager | Progress, including `teacher_alpha` telemetry |

Refer to `docker-compose.yml` for full service definitions and environment variables.
