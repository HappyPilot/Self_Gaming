World State Adapter
===================

The world-state adapter builds a lightweight state dictionary by combining
frame latents with sensor payloads (observations, OCR, detections, rewards).

Encoder
-------

`FrameEncoder` downsamples frames and optionally applies a random projection
to produce a cheap latent vector.

Environment variables:

- `WORLD_STATE_FRAME_SIZE` (default: `64`)
- `WORLD_STATE_LATENT_DIM` (default: `128`)
- `WORLD_STATE_SEED` (default: `1337`)
- `WORLD_STATE_NORMALIZE` (default: `1`)

Usage
-----

```python
from world_state.adapter import WorldStateAdapter

adapter = WorldStateAdapter()
state = adapter.build_state(observation=obs_payload, frame=frame, reward=reward)
```

The resulting dict includes:

- `latent` (list of floats or null)
- `objects` / `object_count`
- `text` / `text_count`
- `player` (if present)
- `reward`
