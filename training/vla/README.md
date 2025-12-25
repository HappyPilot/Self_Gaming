VLA Training Skeleton
=====================

This directory contains a minimal off-device training stub for a
vision-to-action policy (imitation baseline + action chunking head).
It is meant to be a starting point, not a production trainer.

Prepare a dataset
-----------------

Convert recorder sessions into a JSONL dataset:

```bash
python -m training.vla.prepare_dataset \
  --dataset-dir /mnt/ssd/datasets \
  --output /mnt/ssd/datasets/vla_samples.jsonl \
  --game-id poe \
  --horizon-sec 0.5 \
  --sensor-window-sec 0.5
```

Each JSONL row includes:

- `frame_path` (JPEG file path)
- `frame_ts`
- `actions` (list of action dicts within the horizon)
- `sensors` (latest payloads per topic within the window)
- `session_id`, `game_id`, `profile`

Train a stub policy
-------------------

```bash
python -m training.vla.train_stub \
  --dataset /mnt/ssd/datasets/vla_samples.jsonl \
  --output-dir /mnt/ssd/models/vla_stub \
  --epochs 3
```

The stub:

- extracts a cheap latent using `world_state.FrameEncoder`
- predicts a fixed-length action chunk (flattened dx/dy/click/key)
- logs loss to MLflow (if available)
- writes a checkpoint `.pt` file

Optional ONNX export:

```bash
python -m training.vla.train_stub \
  --dataset /mnt/ssd/datasets/vla_samples.jsonl \
  --output-dir /mnt/ssd/models/vla_stub \
  --export-onnx
```

Dependencies
------------

- `torch`
- `numpy`
- `opencv-python` (JPEG decode via cv2)
- `mlflow` (optional, for tracking)

MLflow env vars:

- `MLFLOW_TRACKING_URI` (default: http://127.0.0.1:5001)
- `MLFLOW_EXPERIMENT` (default: self_gaming)
- `MLFLOW_RUN_NAME` (optional)

Notes
-----

- This stub is CPU-friendly and intended for x86/off-device training.
- To build TensorRT engines from ONNX, see `tools/build_trt_engine.sh`.
