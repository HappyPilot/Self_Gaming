# NitroGen integration skeleton (MQTT bridge)

This is a minimal, hackable skeleton to integrate NitroGen as a low-level "hands" policy:
- Windows agent: capture game window -> publish frames to MQTT -> consume actions -> drive a virtual gamepad.
- Linux/Jetson agent: subscribe frames -> call NitroGen inference server -> publish actions back.

## Assumptions
- You have an MQTT broker (e.g., Mosquitto) reachable by both Windows and Linux/Jetson.
- You run NitroGen's server (ZeroMQ on port 5555 by default). This repo provides
  `tools/nitrogen/agents/nitrogen_serve.py` as a fp16-aware wrapper around NitroGen.
  and run the agent on Windows games via `python scripts/play.py --process '<game.exe>'`.
  (See MineDojo/NitroGen README.)

## Quick start (Linux/Jetson)
Run these commands from `tools/nitrogen`:
1) Vendor NitroGen repo into `_vendor/NitroGen`
2) Put checkpoint `ng.pt` into `_vendor/ng.pt`
3) Copy `config/nitrogen.env.example` -> `config/nitrogen.env` and set MQTT/NITROGEN_HOST/NITROGEN_PORT (ZeroMQ).
   On Jetson 8GB, keep `NITROGEN_DTYPE=fp16` to reduce memory.
4) `docker compose -f docker-compose.nitrogen.yml up --build`

You can automate steps 1-2 with the fetch script:

```bash
./scripts/fetch_assets.sh
# Optional integrity checks:
#   NITROGEN_NG_PT_SIZE=<bytes> NITROGEN_NG_PT_SHA256=<hex> ./scripts/fetch_assets.sh
```

## Recommended deployment (Jetson GPU + Mac capture)
Run NitroGen server + proxy on the Jetson (GPU) and only capture/input on the Mac.

Jetson (server + proxy):
1) Place NitroGen repo + weights:
   - `~/self-gaming/tools/nitrogen/_vendor/NitroGen`
   - `~/self-gaming/tools/nitrogen/_vendor/ng.pt`
2) Start NitroGen server:
   - `NITROGEN_DTYPE=fp16 python ~/self-gaming/tools/nitrogen/agents/nitrogen_serve.py \`
     `~/self-gaming/tools/nitrogen/_vendor/ng.pt --port 5555`
3) Start the proxy (publishes actions to `act/cmd`):
   - `PYTHONPATH=~/self-gaming/tools/nitrogen/agents \`
     `MQTT_HOST=127.0.0.1 NITROGEN_HOST=127.0.0.1 NITROGEN_PORT=5555 \`
     `TOPIC_FRAME=self_gaming/vision/frame TOPIC_INTENT=policy_brain/cmd TOPIC_ACTION=act/cmd \`
     `python ~/self-gaming/tools/nitrogen/agents/nitrogen_proxy.py`

Mac (capture/input):
1) Capture frames to the Jetson broker:
   - `MQTT_HOST=<JETSON_IP> TOPIC_FRAME=self_gaming/vision/frame python agents/mac_game_agent/mac_capture_input.py`
2) Input choices:
   - NitroGen actions are gamepad-shaped (`lx/ly/btn`), so **use mac_capture_input for input**
     (INPUT_ENABLED=1) and keep `laptop_input_agent.py` paused.
   - If you want to keep `laptop_input_agent.py` for other policies, run mac_capture_input
     with `INPUT_ENABLED=0` (capture-only).
   - To pause NitroGen input from the laptop_input_agent HTTP panel (port 5010),
     set `INPUT_PAUSE_FILE=/tmp/sg_input_pause` for mac_capture_input (same as laptop_input_agent).
     Use a different pause file if you want independent pause control.

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
The NitroGen server uses ZeroMQ (port 5555 by default) and returns a raw action dict.
Check `scripts/serve.py` logs/docs and then adjust:
- `agents/nitrogen_client.py` (host/port + payload)
- `agents/nitrogen_proxy.py:normalize_action()` (schema + button mapping)

Memory notes:
- On Jetson 8GB, start with `NITROGEN_DTYPE=fp16` to reduce memory usage.
- If fp16 causes instability, try `NITROGEN_DTYPE=fp32` (higher memory).
