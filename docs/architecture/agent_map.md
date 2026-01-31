# Agent Map (living notes)

Purpose: quick mental model of the self-gaming stack, data flow, and key knobs.

## Core data flow (runtime)
1) vision -> vision/frame/preview + vision/observation
2) ocr_easy -> ocr_easy/text (radar OCR)
3) ui_region -> ocr/regions (stable OCR ROIs)
4) simple_ocr -> simple_ocr/text (ROI OCR)
5) scene_agent -> scene/state (fusion: objects + OCR + enemy bars + embeddings)
6) teacher_agent -> teacher/action (LLM guidance)
7) policy_agent -> act/cmd (actions)
8) act_agent -> input bridge (mouse/keyboard)

## Key agents
- vision_agent: publishes frames + observation
- ocr_easy_agent: full-frame OCR (paddle/easyocr)
- ui_region_agent: clusters OCR boxes into stable ROIs
- simple_ocr_agent: ROI OCR with tesseract
- scene_agent: fuses objects/ocr/flags/enemy bars/embeddings -> scene/state
- teacher_agent: LLM-driven action proposal (JSON schema)
- policy_agent: mixes teacher + policy + heuristics -> act/cmd
- act_agent: executes inputs
- embedding_guard_agent: in_game flag + pause logic (scene/flags)
- progress_agent: training/progress metrics

## Vision signals used for behavior
- enemy_bars: red health bars (scene_agent derived) => enemy presence
- targets: OCR targets (text + bbox + center)
- prompt_scores: siglip prompt scoring
- flags: in_game, death, dialog, etc.

## LLM / teacher
- Teacher requires JSON actions (teacher_agent).
- Mac runs llama.cpp; Jetson agents call it via configured endpoint.
- Teacher sees: scene summary, targets, controls, recent actions.

## Key env files
- config/jetson.env: main runtime knobs (OCR, policy, enemy bars)
- docker-compose.yml: service wiring and topics

## Known pitfalls
- OCR bbox format: OCR returns xywh; scene expects xyxy -> must normalize.
- NumPy 2.x breaks OpenCV in policy/scene images (pin numpy<2).
- Observation text_zones can wipe OCR zones; avoid overwriting with empty data.

## Useful topics
- scene/state
- act/cmd
- teacher/action
- ocr_easy/text
- simple_ocr/text
- vision/frame/preview
- vision/observation
- vision/objects

## Quick checks
- enemy bars: `docker logs scene_agent | grep "Enemy bars"`
- OCR: `mosquitto_sub -t ocr_easy/text -C 1`
- Targets: `mosquitto_sub -t scene/state -C 1 | jq .targets`
- Actions: `mosquitto_sub -t act/cmd -C 5`
