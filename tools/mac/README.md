# macOS Input Bridge

This folder contains the macOS input bridge used to turn MQTT actions into local keyboard/mouse events.

## Quick start

```bash
JETSON_IP=10.0.0.68 ACT_TOPIC=act/cmd INPUT_CONTROL_TOPIC=act/control \
  python3 /Users/dima/self-gaming/tools/mac/laptop_input_agent.py
```

Open the control panel:
- http://<mac_ip>:5010

## Requirements
- pyautogui
- paho-mqtt
- pyperclip (optional, for paste)

## macOS permissions
Allow the Python binary you use in:
- System Settings -> Privacy & Security -> Accessibility

If you run the bridge via launchd, you must grant Accessibility to that Python.app as well.

## Cursor bounds (recommended)
Clamp the cursor to the game window so it does not escape:

```bash
# Full screen
INPUT_BOUNDS=screen

# Or explicit bounds: left,top,right,bottom
INPUT_BOUNDS=120,80,1880,1050
```

`GAME_BOUNDS` is accepted as an alias for `INPUT_BOUNDS`.

## Auto-pause when game is not visible (optional)
Enable a scene guard that pauses input if the scene is stale, desktop is detected,
or no game signal is present. It resumes automatically when the game is detected.

```bash
INPUT_REQUIRE_GAME=1
INPUT_SCENE_TOPIC=scene/state
INPUT_NO_GAME_PAUSE_SEC=2.0
INPUT_SCENE_STALE_SEC=2.0
INPUT_DESKTOP_PAUSE_SEC=3.0
INPUT_GAME_KEYWORDS=health,hp,mana,inventory,menu,enemy,boss,player
INPUT_DESKTOP_KEYWORDS=finder,trash,dock,apple menu,desktop,wallpaper
INPUT_PLAYER_CONF_MIN=0.25
```

## Auto-pause when front app is not the game (recommended)
Pause input when the foreground app is not the game. This uses `osascript`
and may require macOS Automation permission for "System Events".

```bash
INPUT_REQUIRE_FRONT_APP=1
INPUT_FRONT_APP=Path of Exile
INPUT_FRONT_APP_CHECK_SEC=1.0
INPUT_FRONT_APP_GRACE_SEC=2.0
```

## MQTT control (optional)

```bash
# Send from any MQTT client
mosquitto_pub -h 10.0.0.68 -t act/control -m '{"cmd":"pause"}'
mosquitto_pub -h 10.0.0.68 -t act/control -m '{"cmd":"resume"}'
mosquitto_pub -h 10.0.0.68 -t act/control -m '{"cmd":"panic"}'
```

## Pause via file flag

```bash
touch /tmp/sg_input_pause
rm /tmp/sg_input_pause
```
