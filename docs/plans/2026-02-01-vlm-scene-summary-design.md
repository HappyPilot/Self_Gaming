# VLM Scene Summary Design

## Context
Agents currently rely on OCR text, enemy bars, and targets to form a scene summary. This leads to “chaotic” movement because there is no semantic understanding of what is on screen. We want a VLM-driven summary (llama.cpp on the Mac) that gives the agents structured context, and we want it to influence actions while still respecting safety constraints.

## Goals
- Generate structured VLM summaries from live frames.
- Publish summaries to a new MQTT topic for all agents.
- Inject summaries into `teacher_agent` prompts so actions reflect context.
- Collect 30-minute behavior logs that include VLM context.

## Non-Goals
- Full multi-modal planner or long-term memory rewrite.
- Replacing safety gating on allowed controls.
- Hardcoding game-specific rules.

## Approach
### 1) New `vlm_summary_agent`
- Subscribes to `vision/frame/preview`.
- Decodes frames and calls llama.cpp VLM on the Mac.
- Produces JSON summary:
  ```json
  {
    "game": "...",
    "summary": "...",
    "player_state": "...",
    "enemies": "...",
    "objectives": "...",
    "ui": "...",
    "risk": "low|medium|high",
    "recommended_intent": "...",
    "timestamp": 123.45
  }
  ```
- Publishes to `scene/summary` and mirrors into `mem/store` with key `scene_summary:<scope>`.
- Adaptive cadence (2–3s base, slower if scene diff is low).
- Backoff + stale fallback if VLM is unavailable.
- Logs VLM latency and status to `/mnt/ssd/logs/vlm_summary/events.jsonl`.

### 2) Teacher agent uses VLM summary
- Subscribe to `scene/summary` and store last summary.
- Inject VLM summary into the prompt as a context hint.
- Add prompt rule: if VLM conflicts with OCR/targets, prefer direct sensors.
- Keep safety constraints (allowed keys, enemy gating, respawn logic).

### 3) Monitoring
- Extend 30-min monitor to include VLM summary snapshot + staleness.
- Log `vlm_latency_ms` and `summary_staleness_sec`.
- Correlate actions with VLM intent.

## Error Handling
- Missing VLM: publish stale summary with status.
- JSON parse failures: log and skip update.
- Long latency: reduce call frequency temporarily.

## Testing
- Unit test: VLM summary parsing and publish.
- Integration test: teacher agent includes `scene/summary` in prompt.

## Rollout
1) Implement `vlm_summary_agent` and topic.
2) Wire into teacher agent prompts.
3) Run 30-min monitoring with VLM context.
