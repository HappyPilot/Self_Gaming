# Game-Agnostic Combat + Vision Design

Date: 2026-01-31

## Goals
- Replace naive "nearest enemy" with strategy-based combat logic.
- Make key usage game-specific (no standard key set).
- Improve vision understanding with UI layout masking and OCR-based skills.
- Learn enemy labels per game over time.

## Non-goals
- No hard dependency on a VLM (only OCR + existing vision signals).
- No full RL retraining changes in this iteration.

## Current state (summary)
- Onboarding uses LLM (text only) to build a control profile.
- policy_agent uses allowed_keys and generic enemy detection.
- scene_agent provides enemy_bars when enabled.
- UI layout mask exists but needs stronger input signals.

## Proposed changes
### 1) Control Profile v2 (keys -> meaning)
- LLM returns a structured list of bindings:
  - keys / combos, action, category, purpose, contexts, risk, confidence.
- Profile is stored per game and published in game/schema.
- Only gameplay categories are allowed by default.
- UI/social/system bindings are blocked unless a goal requires them.

### 2) Skill profiler (OCR + LLM)
- New agent reads OCR tooltips for skill slots.
- LLM classifies each skill as AOE/single/mobility/defense.
- Results stored in game schema and memory.

### 3) UI layout mask
- Onboarding computes play_area and HUD candidates.
- HUD mask is reinforced using stable OCR regions.
- policy_agent avoids clicks in HUD and outside play_area.

### 4) Enemy learning per game
- scene_agent learns enemy labels by overlapping object detections
  with enemy_bars (when available).
- Learned labels are added to dynamic enemy keywords.
- policy_agent targets enemies from this dynamic set.

### 5) Combat strategy in policy_agent
- AOE -> target enemy cluster center.
- Single -> target widest enemy bar.
- Boss -> maintain distance + circle movement.
- Low HP -> retreat + defensive key.
- All key actions are filtered by the control profile.

## Data flow changes
- game_onboarding_agent publishes richer control profile in game/schema.
- scene_agent subscribes to game/schema for enemy terms.
- policy_agent consumes game/schema for key groups and UI layout.
- skill_profiler_agent writes to mem/store and game/schema.

## Error handling
- If LLM profile fails: fall back to mouse-only safe profile.
- If skill profiling fails: mark skill as "unknown" and avoid over-optimization.
- If enemy bars absent: use generic enemy labels only.

## Testing
- Unit: profile normalization, binding parsing, enemy overlap stats.
- Integration: game/schema contains control profile v2 + ui_layout.
- Runtime: policy avoids HUD clicks; keys restricted to allowed groups.

## Rollout
- Deploy behind existing env flags (safe defaults).
- Run onboarding once per game to populate profile.
- Re-run 30-min monitor to compare key usage and engagement rates.
