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
- set `ipc: host` (or `ipc: "service:mq"`) on vision and all consumers
- set `shm_size: 1g` (or similar) to avoid random SHM failures
