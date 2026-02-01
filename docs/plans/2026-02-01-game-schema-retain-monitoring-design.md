# Game Schema Retain + 30-Min Monitor Design

## Context
Agents rely on `game/schema` to learn the control profile, UI layout, and allowed keys. Today the onboarding agent publishes the schema without MQTT retain, so late subscribers miss the latest profile unless onboarding is re-run. We also need a short monitoring pass (30 minutes) with screenshots to assess action quality and engagement.

## Goals
- Ensure any agent that connects later immediately receives the last known `game/schema`.
- Run a 30-minute monitor on Jetson that records action/scene correlation and sampled screenshots.
- Produce a compact summary: key-press rate, enemy-bar hit rate, OCR-target click rate.

## Non-Goals
- Building a long-lived monitoring service.
- Changing schema content or control profile semantics.
- Introducing new agent dependencies.

## Proposed Changes
### 1) Retained game/schema publish
- In `game_onboarding_agent`, publish `game/schema` with `retain=True`.
- Keep `mem/store` writes unchanged to preserve existing memory flow.

### 2) One-off 30-minute monitor script on Jetson
- Subscribe to `act/cmd` and `scene/state`.
- Cache the latest `scene/state` snapshot for action correlation.
- Sample screenshots from `vision/frame/preview` every 30–60 seconds via `utils.frame_transport.get_frame_bytes`.
- Emit `events.jsonl` with timestamps, action type, target_norm, and scene summary (enemy bars + OCR targets).
- Emit `summary.json` with counts, rates, and error totals.

## Data Flow
1) `game_onboarding_agent` publishes `game/schema` with retain.
2) Agents (policy/act/scene) subscribe and immediately receive the schema on connect.
3) Monitor listens to `act/cmd` and `scene/state`, correlates actions with latest scene snapshot.
4) Monitor writes JSONL + summary + screenshots to `/mnt/ssd/logs/monitor_<timestamp>/`.

## Metrics
- `key_press_rate`: key_press count / total actions.
- `enemy_bar_hit_rate`: action targets inside enemy-bar boxes / total click actions.
- `ocr_target_click_rate`: action targets inside OCR target boxes / total click actions.

## Error Handling
- Missing scene/state: log and skip correlation for that action.
- Frame decode failure: log and continue; do not abort.
- JSON parse errors: increment error counters; keep monitor running.

## Testing
- Unit test: verify `game/schema` publish uses `retain=True` (TDD: red then green).
- Smoke test: 60-second dry run on Jetson to confirm logs and screenshots.
- Full run: 30-minute monitor to collect metrics and qualitative samples.

## Rollout
- Implement retain change, run unit tests.
- Deploy to Jetson, run 60-second dry run, then the 30-minute session.
