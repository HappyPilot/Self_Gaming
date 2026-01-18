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

`policy_agent.py` now blends its heuristic actions with the teacher's recommendations. A linear annealing coefficient `teacher_alpha` (configurable via `TEACHER_ALPHA_START` and `TEACHER_ALPHA_DECAY_STEPS`) starts at 1.0, prioritising teacher advice, then decays to rely on the learned policy. The final action is emitted on both `control/keys` and `act/cmd`, preserving the previous behaviour.
Set `POLICY_LAZY_LOAD=1` (default) to load policy weights on first observation instead of at startup (use `0` to preload). Use `POLICY_LAZY_RETRY_SEC` to retry lazy-load after a failure (default 120s).

## Shared Strategy State (Optional)

Run the shared-state server:

```bash
python -m shared_state.state_server
```

Then point agents to it:

```bash
export STRATEGY_STATE_HOST=127.0.0.1
export STRATEGY_STATE_PORT=54001
export STRATEGY_STATE_AUTHKEY=strategy
```

If using Docker Compose, enable the optional `strategy_state` service (profile `shared_state`). Use `STRATEGY_STATE_HOST=strategy_state` for bridged networking or `127.0.0.1` when running other services with host networking.

## Object-Detection Agent

- `object_detection_agent.py` subscribes to `vision/frame/preview` by default (set `VISION_FRAME_TOPIC=vision/frame/full` for full quality), runs a pluggable YOLO backend (Ultralytics by default), and emits structured detections on `vision/objects`.
- The agent is packaged in its own container (`local/object-detection-agent`). Drop TensorRT/Ultralytics weights under `/mnt/ssd/models/yolo/<checkpoint>` and point `OBJECT_MODEL_PATH` there.
- Relevant environment switches (see `docker-compose.yml`):
  - `VISION_FRAME_INTERVAL` / `VISION_FRAME_JPEG_QUALITY` - sampling rate + encoding coming from `vision_agent`.
  - `OBJECT_DETECTOR_BACKEND` - `onnx`, `ultralytics` (`torch`/`trt` aliases), or `dummy` for smoke tests. `torch` expects `.pt` weights, `onnx` expects a `.onnx`, and `trt`/`tensorrt` expects a `.engine`.
  - `OBJECT_CONF_THRESHOLD`, `OBJECT_IOU_THRESHOLD`, `OBJECT_QUEUE` - runtime tuning knobs.
  - `OBJECT_FALLBACK_*` - optional secondary detector used when the primary returns zero boxes (same keys as the primary with the `OBJECT_FALLBACK_` prefix).

Example: YOLO-World TensorRT primary + ONNX fallback:

```bash
export OBJECT_DETECTOR_BACKEND=tensorrt
export OBJECT_MODEL_PATH=/mnt/ssd/models/yolo/yolov8s_worldv2_320_fp16.engine
export OBJECT_CLASS_PATH=/mnt/ssd/models/yolo/coco.names
export OBJECT_CONF_THRESHOLD=0.03
export OBJECT_FALLBACK_BACKEND=onnx
export OBJECT_FALLBACK_MODEL_PATH=/mnt/ssd/models/yolo/yolo11n_416_fp32.onnx
export OBJECT_FALLBACK_CLASS_PATH=/mnt/ssd/models/yolo/classes_generic.txt
export OBJECT_FALLBACK_CONF_THRESHOLD=0.1
```

The `scene_agent` now fuses OCR, mean luminance, and the most recent detection payload so downstream agents receive `objects` with `(class, confidence, box)` triples in every `scene/state` update.

## DeepStream Perception Bridge

`perception_ds` publishes `vision/observation` with `detections` (xywh boxes + `class_id`). The `scene_agent` now converts those into `yolo_objects` so `scene/state` receives usable objects even when the Python perception agent is down.

Controls:
- `SCENE_CLASS_PATH` (or `YOLO_CLASS_PATH`) to map `class_id` to labels.
- `ENGINE_INPUT_SIZE` to normalize DeepStream boxes when frame dimensions are not present.
- `OBJECT_PREFER_OBSERVATION=1` keeps DeepStream detections as the primary source; `OBJECT_FALLBACK_AFTER_SEC` controls how long to wait before accepting `vision/objects` as fallback.

## Perception Agent (YOLO + OCR)

`perception_agent.py` builds `vision/observation` (consumed by `scene_agent` for `scene/state`). It uses `DETECTOR_BACKEND` + the `YOLO11_*` settings, not the `OBJECT_*` settings.
This GPU-backed path is experimental; enable it via the `experimental` docker-compose profile if needed.

Relevant environment switches:
- `DETECTOR_BACKEND` - `yolo11_torch`, `yolo11_trt`, `yolo_trt_engine`, or `yoloworld`.
- `YOLO11_WEIGHTS` - `.pt` or `.engine` path for the selected backend.
- `YOLO11_CONF`, `YOLO11_IMGSZ` - detection thresholds and input size.
- `YOLO_CLASS_LIST` / `YOLO_CLASS_PATH` - class names for `yolo_trt_engine` (comma list or a newline file).
- `YOLO_WORLD_CLASSES` - class prompts for the `yoloworld` backend.

Example: YOLO-World TensorRT engine for perception:

```bash
export DETECTOR_BACKEND=yolo_trt_engine
export YOLO11_WEIGHTS=/mnt/ssd/models/yolo/yolov8s_worldv2_320_fp16.engine
export YOLO11_IMGSZ=320
export YOLO11_CONF=0.03
export YOLO_CLASS_PATH=/mnt/ssd/models/yolo/coco.names
```

## Visual Embeddings (SigLIP2 / V-JEPA)

`vl_jepa_agent.py` subscribes to `vision/frame/preview` and publishes visual embeddings on `vision/embeddings`.
You can run it with lightweight SigLIP2 weights (no NitroGen required):

```bash
export VL_JEPA_BACKEND=siglip2
export VL_JEPA_MODEL_ID=google/siglip2-base-patch16-224
export VL_JEPA_INPUT_SIZE=224
export VL_JEPA_EMBED_DIM=768
export VL_JEPA_DEVICE=cpu
export VL_JEPA_FP16=0
export VL_JEPA_ATTN_IMPL=sdpa   # optional
export VL_JEPA_EMBED_FPS=2      # rate-limit embedding publishes
export VL_JEPA_CACHE_HASH=1     # reuse last embedding if frame bytes repeat
```

This keeps the control loop and MQTT contracts intact while giving your agents a strong
pretrained vision backbone. If you prefer existing TorchScript/TensorRT encoders,
keep `VL_JEPA_BACKEND=torchscript` or `tensorrt` and set the corresponding paths.

To feed embeddings into policy + training, enable:

```bash
export EMBED_FEATURE_ENABLED=1
export EMBED_FEATURE_DIM=128
export EMBED_FEATURE_SOURCE_DIM=768
```

## Embedding Guard (In-Game Gate)

`embedding_guard_agent.py` listens to `vision/embeddings`, compares them to stored centroids, and publishes `scene/flags` with `in_game=true/false`. This gates actions and teacher updates when the screen is not in-game.

Quick setup:
- Collect samples by running with `EMBEDDING_GUARD_MODE=collect_game` and `collect_non`.
- Store centroids under `/mnt/ssd/models/embedding_guard`.
- Enable gating with `POLICY_REQUIRE_IN_GAME=1` and `TEACHER_REQUIRE_IN_GAME=1`.

## Reward Manager

`reward_manager.py` pulls three signals to publish dense rewards on `train/reward`:

1. Tail the local PoE client log at `/mnt/ssd/poe_client.txt` (streamed from Windows via the Flask receiver) and normalise entries such as "You have entered...", "You have killed...", and "Picked up ...".
2. Consume `scene/state` so the latest YOLO detections contribute enemy density + loot pressure (important for map-progress/loot weights).
3. Watch `act/result` to stay in lockstep with the control loop.

The calculator mirrors the curriculum in `poe_reward_system_v1.md`: stage-specific weights (`S0...S4`), step-costs, death penalties, and loot-value buckets (currency, maps, contracts, etc.). Rewards are clipped to `[-1, 1]` and include a `components` breakdown plus the last few parsed events for debugging.

## Training Pipeline Updates

- `recorder_agent.py` writes session datasets under `RECORDER_DATASET_DIR/<game>/<session_id>/` (frames, actions.jsonl, sensors.jsonl, meta.json, qc.json; see `recording/README.md`) and keeps legacy `sample_*.json` files in `RECORDER_DIR`.
- `teach_agent.py` includes the distillation schedule plus the reward topic when it emits `train/jobs`.
- `train_manager_agent.py` consumes the richer samples, applies a KL distillation loss against the teacher actions, and adds a reward-weighted behaviour-cloning/RL loss (tunable via `TEACHER_KL_WEIGHT`, `REWARD_WEIGHT`).

### Workflow

1. Ensure all agents are running (including `teacher_agent` and `reward_manager`).
2. During early training epochs, the policy will follow the teacher's commands; as `teacher_alpha` decays, it transitions to autonomous behaviour.
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
  - `LOG_DIR` - preferred directory (falls back automatically if unavailable).
  - `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` - rotation settings (default 5 MB x 5 files).
  - `LOG_LEVEL` - default `INFO`; set to `DEBUG` for chatty traces.

- To copy Jetson logs onto this Mac, run `tools/sync_jetson_logs.sh` (uses `rsync`).  Override `JETSON_HOST`, `JETSON_LOG_DIR`, or `JETSON_LOGS_DEST` if the defaults (`user@jetson.local:/mnt/ssd/logs` -> `logs_jetson/`) need to change.

### Log Monitor Agent

`log_monitor_agent.py` tails every `*.log` file and publishes:

| Topic | Payload |
| --- | --- |
| `logs/summary` | Aggregate counts per agent (`lines`, `warnings`, `errors`). |
| `logs/alerts` | Structured alerts when errors, tracebacks, failed jobs, or NaN losses are detected (rate-limited by `LOG_ALERT_COOLDOWN`, default 30 s). |

Configuration knobs:

- `LOG_MONITOR_DIRS` (colon-separated) or `LOG_MONITOR_DIR` - directories to scan (default `/app/logs`).
- `LOG_MONITOR_INTERVAL` - scan cadence in seconds.
- `LOG_MONITOR_TRAINERS` - comma-separated list of log sources that should trigger the extra training-health heuristics (`train_manager,policy_agent,teach_agent` by default).

Add `log_monitor` to the `docker-compose.yml` stack (already wired) and check `logs/alerts` for actionable messages when training stalls or errors recur.

## MQTT Topics

| Topic | Producer | Purpose |
| --- | --- | --- |
| `teacher/action` | Teacher agent | Natural-language action suggestions + reasoning |
| `train/reward` | Reward manager | Scalar reward events consumed by the trainer |
| `act/cmd` / `control/keys` | Policy agent | Final blended control command |
| `train/status` | Train manager | Progress, including `teacher_alpha` telemetry |

Refer to `docker-compose.yml` for full service definitions and environment variables.
