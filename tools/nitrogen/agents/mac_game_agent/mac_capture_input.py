from __future__ import annotations

import atexit
import base64
import io
import json
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt
import pyautogui
from mss import mss
from PIL import Image


@dataclass(frozen=True)
class Cfg:
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str
    mqtt_pass: str

    capture_fps: float
    capture_monitor: int
    capture_region: Optional[Tuple[int, int, int, int]]
    resize_w: int
    resize_h: int
    jpeg_quality: int

    topic_frame: str
    topic_action: str

    mouse_range: float
    mouse_deadzone: float
    button_map: Dict[str, str]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _parse_region(raw: str) -> Optional[Tuple[int, int, int, int]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return None
    try:
        left, top, width, height = [int(p) for p in parts]
    except Exception:
        return None
    return left, top, width, height


def _parse_button_map(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, action = token.split("=", 1)
        key = key.strip().upper()
        action = action.strip().lower()
        if key:
            out[key] = action
    return out


def load_cfg() -> Cfg:
    return Cfg(
        mqtt_host=_env("MQTT_HOST", "localhost"),
        mqtt_port=int(_env("MQTT_PORT", "1883")),
        mqtt_user=_env("MQTT_USER", ""),
        mqtt_pass=_env("MQTT_PASS", ""),
        capture_fps=float(_env("CAPTURE_FPS", "15")),
        capture_monitor=int(_env("CAPTURE_MONITOR", "1")),
        capture_region=_parse_region(_env("CAPTURE_REGION", "")),
        resize_w=int(_env("RESIZE_W", "256")),
        resize_h=int(_env("RESIZE_H", "256")),
        jpeg_quality=int(_env("JPEG_QUALITY", "70")),
        topic_frame=_env("TOPIC_FRAME", "self_gaming/vision/frame"),
        topic_action=_env("TOPIC_ACTION", "self_gaming/control/action"),
        mouse_range=float(_env("MOUSE_RANGE", "40")),
        mouse_deadzone=float(_env("MOUSE_DEADZONE", "0.1")),
        button_map=_parse_button_map(
            _env(
                "BUTTON_MAP",
                "A=click_primary,B=click_secondary,X=key_space,Y=key_e,"
                "LB=key_q,RB=key_w,START=key_tab,BACK=key_esc",
            )
        ),
    )


def img_to_jpeg_b64(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _apply_button(action: str, pressed: bool, held: set[str]) -> None:
    if action in {"click_primary", "click_secondary", "click_middle"}:
        button = (
            "left"
            if action == "click_primary"
            else "right"
            if action == "click_secondary"
            else "middle"
        )
        if pressed and action not in held:
            pyautogui.mouseDown(button=button)
            held.add(action)
        elif not pressed and action in held:
            pyautogui.mouseUp(button=button)
            held.discard(action)
        return

    if action.startswith("key:"):
        key = action.split(":", 1)[1]
    elif action.startswith("key_"):
        key = action.split("_", 1)[1]
    else:
        key = action

    if not key:
        return
    if pressed and key not in held:
        pyautogui.keyDown(key)
        held.add(key)
    elif not pressed and key in held:
        pyautogui.keyUp(key)
        held.discard(key)


def main() -> None:
    cfg = load_cfg()

    pyautogui.FAILSAFE = os.getenv("PYAUTOGUI_FAILSAFE", "1") != "0"
    pyautogui.PAUSE = float(os.getenv("PYAUTOGUI_PAUSE", "0"))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if cfg.mqtt_user:
        client.username_pw_set(cfg.mqtt_user, cfg.mqtt_pass)
    client.connect(cfg.mqtt_host, cfg.mqtt_port, keepalive=30)

    last_action: Dict[str, Any] = {"lx": 0, "ly": 0, "rx": 0, "ry": 0, "btn": {}}
    held_buttons: set[str] = set()

    def cleanup() -> None:
        for item in list(held_buttons):
            try:
                if item in {"click_primary", "click_secondary", "click_middle"}:
                    button = (
                        "left"
                        if item == "click_primary"
                        else "right"
                        if item == "click_secondary"
                        else "middle"
                    )
                    pyautogui.mouseUp(button=button)
                else:
                    pyautogui.keyUp(item)
            except Exception:
                pass
            held_buttons.discard(item)

    def _handle_signal(_signum, _frame) -> None:
        raise SystemExit()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def on_action(_c, _u, msg):
        nonlocal last_action
        try:
            last_action = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            pass

    client.subscribe(cfg.topic_action, qos=0)
    client.on_message = on_action
    client.loop_start()

    sct = mss()

    dt = 1.0 / max(cfg.capture_fps, 1.0)

    while True:
        t0 = time.time()

        if cfg.capture_region:
            left, top, width, height = cfg.capture_region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        else:
            monitor = sct.monitors[max(1, min(cfg.capture_monitor, len(sct.monitors) - 1))]

        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img_small = img.resize((cfg.resize_w, cfg.resize_h))
        jpeg_b64 = img_to_jpeg_b64(img_small, quality=cfg.jpeg_quality)

        frame_msg = {
            "ts": time.time(),
            "source": "mac",
            "resize": [cfg.resize_w, cfg.resize_h],
            "img_b64_jpg": jpeg_b64,
        }
        client.publish(cfg.topic_frame, json.dumps(frame_msg), qos=0)

        lx = clamp(float(last_action.get("lx", 0.0)), -1.0, 1.0)
        ly = clamp(float(last_action.get("ly", 0.0)), -1.0, 1.0)
        if abs(lx) >= cfg.mouse_deadzone or abs(ly) >= cfg.mouse_deadzone:
            dx = int(lx * cfg.mouse_range)
            dy = int(-ly * cfg.mouse_range)
            if dx or dy:
                pyautogui.moveRel(dx, dy, duration=0)

        btn = last_action.get("btn", {}) or {}
        for name, mapped in cfg.button_map.items():
            pressed = bool(btn.get(name, 0))
            _apply_button(mapped, pressed, held_buttons)

        spent = time.time() - t0
        if spent < dt:
            time.sleep(dt - spent)


if __name__ == "__main__":
    main()
