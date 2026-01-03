# Quickstart (Demo Compose)

This demo stack runs without a camera or GPU. It publishes a tiny demo frame to MQTT, builds a minimal scene update, emits stub actions, and logs the action stream.

## Run

```bash
docker compose -f docker-compose.demo.yml up --build
```

Requires `docker-compose.demo.yml` (added in Phase1 PR04).

If port 1883 is already in use, change the port mapping in `docker-compose.demo.yml` (for example, `1884:1883`).

## Verify

Look for output like:
- `demo_frame_source` reporting ticks and mean values
- `demo_policy` emitting actions
- `demo_action_sink` printing `act/cmd` payloads

To follow only the action stream:

```bash
docker compose -f docker-compose.demo.yml logs -f demo_action_sink
```

## Stop

```bash
docker compose -f docker-compose.demo.yml down
```

## Customize

Environment overrides (add to your shell or compose override):
- `FRAME_INTERVAL` - seconds between demo frames (default 1.0)
- `DEMO_ACTIONS` - comma-separated actions (default `noop,click,move,key`)
- `DEMO_ACTION_MIN_INTERVAL` - throttle action publishing (default 0.5)
- `DEMO_IMAGE_B64` - base64-encoded image payload (default 1x1 PNG)
- `DEMO_IMAGE_WIDTH` / `DEMO_IMAGE_HEIGHT` - image size metadata
- `VISION_FRAME_TOPIC` - defaults to preview; set to `vision/frame/full` when you need full-quality frames
- `FRAME_TRANSPORT` - `mqtt` (base64) or `shm` (shared memory descriptor)

For FRAME_TRANSPORT=shm, containers must share an IPC namespace and have enough /dev/shm:
- set `ipc: host` (or `ipc: "service:mq"`) on vision and every consumer in the pipeline (same IPC namespace, or you will see "SHM segment not found")
- set `shm_size: 1g` (or similar) to avoid random SHM failures

### Optional: enable Titans policy adapter

Titans support is optional. Build the policy image with Titans enabled:

```bash
docker build -f docker/policy/Dockerfile --build-arg WITH_TITANS=1 -t sg-policy:with-titans .
```

Default build (no Titans):

```bash
docker build -f docker/policy/Dockerfile -t sg-policy:no-titans .
```

Jetson compose build (ensures WITH_TITANS is passed as a build arg):

```bash
WITH_TITANS=1 docker compose --env-file config/jetson.env build policy
docker compose --env-file config/jetson.env up -d
```

Note: `env_file` controls runtime env for containers. Build args come from your shell or `--env-file`, so set
`WITH_TITANS=1` explicitly (or export it) when building the policy image.

Local install (requires torch already installed):

```bash
python3 -m pip install -r requirements-titans.txt
```

Titans environment variables:
- `TITANS_DIM` - default 256
- `TITANS_CHUNK` - default 32
- `TITANS_ALLOW_PROJECTOR` - default 0
- `TITANS_LOAD_MEMORY` - default 0
- `TITANS_MEM_PATH` - default `titans_memory.pth`
- `TITANS_DEVICE` - default auto cuda/cpu
- `TITANS_UPDATE_INTERVAL` - default 1 (update memory every N ticks)

Operational notes:
- Titans backend is experimental.
- On Jetson 8 GB keep `TITANS_DIM=256` unless you have measurements.
- When `TITANS_UPDATE_INTERVAL>1`, the first tick may return a fallback until the buffer is filled.
- If behavior drifts or degrades, remove the memory file at `TITANS_MEM_PATH`.

Sanity checks:

A) Docker sanity (policy image + container):

```bash
WITH_TITANS=1 docker compose --env-file config/jetson.env build policy
docker compose --env-file config/jetson.env up -d
```

B) Local sanity (host unittest after install):

```bash
python3 -m pip install -r requirements-titans.txt
python3 tests/test_policy_titans_adapter.py
```

Note: `unittest` runs on the host and requires torch installed.
