# NitroGen integration skeleton (MQTT bridge)

This is a minimal, hackable skeleton to integrate NitroGen as a low-level "hands" policy:
- Windows agent: capture game window -> publish frames to MQTT -> consume actions -> drive a virtual gamepad.
- Linux/Jetson agent: subscribe frames -> call NitroGen inference server -> publish actions back.

## Assumptions
- You have an MQTT broker (e.g., Mosquitto) reachable by both Windows and Linux/Jetson.
- You run NitroGen's server as described in their repo: `python scripts/serve.py <path_to_ng.pt>`
  and run the agent on Windows games via `python scripts/play.py --process '<game.exe>'`.
  (See MineDojo/NitroGen README.)

## Quick start (Linux/Jetson)
Run these commands from `tools/nitrogen`:
1) Vendor NitroGen repo into `_vendor/NitroGen`
2) Put checkpoint `ng.pt` into `_vendor/ng.pt`
3) Copy `config/nitrogen.env.example` -> `config/nitrogen.env` and set MQTT/NITROGEN_BASE_URL
4) `docker compose -f docker-compose.nitrogen.yml up --build`

## Quick start (Windows)
Run these commands from `tools/nitrogen`:
1) `pip install -r agents/win_game_agent/requirements.txt`
2) Set env vars:
   - MQTT_HOST (broker IP)
   - GAME_PROCESS (YourGame.exe)
3) `python agents/win_game_agent/win_capture_input.py`

## Quick start (macOS)
Run these commands from `tools/nitrogen`:
1) `pip install -r agents/mac_game_agent/requirements.txt`
2) Set env vars:
   - MQTT_HOST (broker IP)
   - CAPTURE_MONITOR=1 (or CAPTURE_REGION=left,top,width,height)
   - INPUT_ENABLED=0 (capture-only; use laptop_input_agent.py for input)
   - INPUT_PAUSE_FILE=/tmp/sg_input_pause (shared pause toggle)
3) `python agents/mac_game_agent/mac_capture_input.py`

Notes:
- Run the game agent on the same host as the game (Windows or macOS).
- Both Windows and macOS agents publish frames to TOPIC_FRAME and consume actions from TOPIC_ACTION.
- NitroGen's official play agent is Windows-only; the macOS agent here is a custom MQTT capture+input wrapper.

Safety:
- `PYAUTOGUI_FAILSAFE=1` (default) keeps the PyAutoGUI failsafe enabled.
- Set `PYAUTOGUI_FAILSAFE=0` only if you understand the risks.

## IMPORTANT: Endpoint + action mapping
The NitroGen server's HTTP endpoint and output JSON schema may differ from this skeleton.
Check `scripts/serve.py` logs/docs and then adjust:
- `agents/nitrogen_client.py` (route + payload)
- `agents/nitrogen_proxy.py:normalize_action()` (schema + button mapping)
