# System Map (Self-Gaming)

## Purpose
Self-Gaming is a multi-agent stack that observes a game screen, extracts signals
(vision, OCR, prompts, enemy bars), fuses them into a scene state, and chooses
actions via policy and teacher agents. It runs on Jetson for vision and uses
llama.cpp on the Mac for LLM calls.

## Core data flow (runtime)
1. vision_agent -> vision/frame/preview + vision/frame/full + vision/mean
2. ocr_easy_agent -> ocr_easy/text (full-frame OCR)
3. ui_region_agent -> ocr/regions (stable UI text ROIs)
4. simple_ocr_agent -> simple_ocr/text (ROI OCR)
5. scene_agent -> scene/state (objects + OCR + enemy bars + prompts)
6. teacher_agent -> teacher/action (LLM guidance)
7. policy_agent -> act/cmd (action selection)
8. act_agent -> input bridge (mouse/keyboard)

## Key agents and roles
- vision_agent: captures frames from V4L2 or HTTP source
- perception_ds / object_detection_agent: object detection pipeline
- ocr_easy_agent: full-frame OCR (radar mode)
- ui_region_agent: clusters OCR boxes into stable UI regions
- simple_ocr_agent: fast OCR on stable regions
- scene_agent: fuses objects, OCR, enemy bars, embeddings -> scene/state
- embedding_guard: in_game detection and pause logic
- teacher_agent: LLM guidance (game-aware suggestions)
- policy_agent: selects actions and enforces safety rules
- act_agent: applies actions and logs results
- game_onboarding_agent: builds game schema, control profile, UI layout
- progress_agent: health + understanding metrics

## MQTT topics (most used)
- vision/frame/preview, vision/frame/full, vision/mean
- ocr_easy/text, ocr/regions, simple_ocr/text
- scene/state, scene/flags
- teacher/action, act/cmd, act/result
- game/schema, game/identity
- progress/status, progress/understanding

## Configuration sources
- config/defaults.env
- config/jetson.env (profile overrides)
- config/mac.env (if used)
- data/control_profiles.json (per-game keys + semantics)
- data/prompt_profiles.json (vision prompt sets)

## Onboarding and game schema
- game_onboarding_agent publishes game/schema containing:
  - ui_layout (play_area + HUD candidates)
  - controls (probed inputs)
  - profile (per-game control profile)
  - signals (danger/progress tokens)
- policy_agent and act_agent use game/schema to restrict actions

## LLM usage
- llama.cpp runs on the Mac, accessed via LLM_ENDPOINT
- LLM calls are gated via LLM_GATE_FILE to avoid overload
- Onboarding can request control profile and prompt profile

## Safety posture
- policy_agent uses UI layout to avoid HUD clicks
- act_agent enforces allowed_keys from the control profile
- forbidden text and forbidden keys suppress risky actions

## Key docs
- docs/architecture/agent_map.md
- docs/mqtt/topics.md
- PROJECT_OVERVIEW.md (auto-generated)
