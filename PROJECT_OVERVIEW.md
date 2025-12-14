# Project Overview (auto-generated)
            Generated: 2025-12-14T08:56:16
            Git: 80ce31c

            ## What this repo is
            Self-Gaming multi-agent stack (vision/OCR/policy/act) with Jetson + DeepStream + MQTT.

            ## Top-level directories
            - `agents/`
- `data/`
- `docker/`
- `docs/`
- `logs/`
- `logs_jetson/`
- `models/`
- `multiagent/`
- `scripts/`
- `services/`
- `tests/`
- `tools/`

            ## Docker Compose services (current)
            ```text
            /bin/sh: docker: command not found
            ```

            ## Compose config snapshot
            Saved to: `docs/compose_config.yaml`

            ## MQTT topics discovered (best-effort scan)
            - `act/cmd`
- `act/feedback`
- `act/log`
- `act/request`
- `act/result`
- `application/json`
- `auto_train/status`
- `budget/summary`
- `budget/update`
- `codex/cmd`
- `codex/reply`
- `control/keys`
- `cursor/state`
- `data/poe_yolo`
- `demonstrator/status`
- `eval/cmd`
- `eval/report`
- `eval/request`
- `eval/result`
- `game/schema`
- `goals/high_level`
- `goap/tasks`
- `google/owlvit-base-patch32`
- `logs/alerts`
- `logs/summary`
- `mem/query`
- `mem/reply`
- `mem/response`
- `mem/store`
- `n/a`
- `ocr/cmd`
- `ocr/regions`
- `ocr/status`
- `ocr/text`
- `ocr/unified`
- `ocr_easy/cmd`
- `ocr_easy/text`
- `policy_brain/cmd`
- `policy_brain/metrics`
- `progress/status`
- `recorder/status`
- `replay/sample-request`
- `replay/sample-response`
- `replay/store`
- `research/events`
- `scene/state`
- `sim_core/action`
- `sim_core/cmd`
- `sim_core/state`
- `simple_ocr/text`
- `system/alert`
- `system/health`
- `system/thermal`
- `teach/cmd`
- `teach/request`
- `teacher/action`
- `teacher/status`
- `train/checkpoints`
- `train/jobs`
- `train/reward`
- `train/status`
- `vision/cmd`
- `vision/config`
- `vision/frame`
- `vision/hints`
- `vision/mean`
- `vision/objects`
- `vision/obs`
- `vision/observation`
- `vision/snapshot`
- `vision/status`
- `world_model/pred_error`

            ## Data flow (expected)
            Video -> Vision/DeepStream -> OCR (radar+ROI) -> Scene aggregation -> Policy/Teacher -> Act bridge.

            ## Modes
            - `DS_SOURCE=v4l2` uses `/dev/videoX` capture; HTTP viewer optional.
            - OCR radar: low-frequency Paddle via ocr_easy; ROI OCR via simple_ocr; unified stream `ocr/unified`.
            - Policy brain: shadow mode publishes `policy_brain/cmd`, metrics on `policy_brain/metrics`.

            ## Troubleshooting
            - V4L2 device mismatch: check `/dev/video*`, set `DS_V4L2_DEVICE`, restart perception_ds.
            - DeepStream restarts: verify `DS_SOURCE` and caps, see `docker logs perception_ds`.
            - OCR high CPU: ensure thread caps env, radar interval, and frame-hash gating.
            - GPU idle: ensure pgie running and GR3D_FREQ > 0 in `tegrastats`.

            ## Health snapshot commands
            - `free -h; swapon --show`
            - `timeout 10s tegrastats`
            - `docker stats --no-stream`
            - `docker compose ps`
