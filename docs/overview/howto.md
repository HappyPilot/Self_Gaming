# How-To (Self-Gaming)

## Start the stack (Jetson)
```bash
docker compose up -d
```

## Run onboarding (once per game)
```bash
docker compose run --rm onboarding
```
Notes:
- Onboarding publishes game/schema and stores a control profile.
- The profile is later used by policy_agent and act_agent.
- Gameplay keys are allowed by default; UI/social keys are blocked unless explicitly needed.

## Check main topics
```bash
mosquitto_sub -t scene/state -C 1
mosquitto_sub -t teacher/action -C 1
mosquitto_sub -t act/cmd -C 5
```

## Check OCR and UI regions
```bash
mosquitto_sub -t ocr_easy/text -C 1
mosquitto_sub -t ocr/regions -C 1
mosquitto_sub -t simple_ocr/text -C 1
```

## Skill profiler (tooltip -> type)
```bash
mosquitto_pub -t skill_profiler/cmd -m '{\"slot\":\"q\",\"text\":\"Deals damage in an area around you\"}'
mosquitto_sub -t skill_profiler/result -C 1
```

## Check enemy bars
```bash
docker logs scene_agent | rg -n "Enemy bars"
```

## Verify vision source
```bash
mosquitto_sub -t vision/frame/preview -C 1 >/dev/null && echo frame_ok
```

## Update config
- defaults: config/defaults.env
- Jetson overrides: config/jetson.env
- Per-game controls: data/control_profiles.json

## Common fixes
- No frames: verify VIDEO_DEVICE or VIDEO_SOURCE and V4L2 permissions.
- OCR too heavy: reduce OCR interval and keep thread limits at 1.
- DeepStream restarts: ensure DS_SOURCE and DS_V4L2_DEVICE are correct.

## Useful commands
```bash
docker compose ps
Docker compose logs --tail 80 scene
```
