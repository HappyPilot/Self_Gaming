# Embeddings-First Vision (Jetson)

## Goal
Make VL-JEPA embeddings the primary state signal for policy, with minimal compute. Disable heavy perception (YOLO/OCR) by default.

## Approach
- Keep `vision` + `vl_jepa_agent` + `siglip_prompt_agent` + `scene_agent` + `embedding_guard_agent` + `policy_agent`.
- Remove OCR/YOLO dependencies from `scene_agent` startup.
- Gate actions on fresh embeddings + in-game flag.
- Disable enemy-bar heuristics and OCR-target usage in policy.

## Compose changes
- Add `profiles: [heavy]` to OCR + perception services (ocr, ocr_easy, simple_ocr, ui_region, perception, perception_ds).
- Remove OCR services from `scene` depends_on; keep `vl_jepa` and `siglip_prompt` as deps.
- Remove `ocr_easy` from `policy_brain` depends_on.

## Env changes (jetson)
- `SCENE_ENEMY_BAR_ENABLE=0`
- `POLICY_USE_OCR_TARGETS=0`

## Expected outcome
Default stack runs embeddings-first and avoids heavyweight perception unless `COMPOSE_PROFILES=heavy` is enabled.
