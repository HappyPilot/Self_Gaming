from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt
import psutil
import win32gui
import win32process
from mss import mss
from PIL import Image

import vgamepad as vg


@dataclass(frozen=True)
class Cfg:
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str
    mqtt_pass: str

    process_name: str  # 'game.exe'
    capture_fps: float

    topic_frame: str
    topic_action: str


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def load_cfg() -> Cfg:
    return Cfg(
        mqtt_host=_env("MQTT_HOST", "localhost"),
        mqtt_port=int(_env("MQTT_PORT", "1883")),
        mqtt_user=_env("MQTT_USER", ""),
        mqtt_pass=_env("MQTT_PASS", ""),
        process_name=_env("GAME_PROCESS", "game.exe"),
        capture_fps=float(_env("CAPTURE_FPS", "15")),
        topic_frame=_env("TOPIC_FRAME", "self_gaming/vision/frame"),
        topic_action=_env("TOPIC_ACTION", "self_gaming/control/action"),
    )


def find_hwnd_by_process_name(exe_name: str) -> Optional[int]:
    pids = [
        p.pid
        for p in psutil.process_iter(attrs=["name"])
        if (p.info["name"] or "").lower() == exe_name.lower()
    ]
    if not pids:
        return None
    target = set(pids)
    found: Optional[int] = None

    def enum_cb(hwnd, _):
        nonlocal found
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid in target:
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            if w > 200 and h > 200:
                found = hwnd

    win32gui.EnumWindows(enum_cb, None)
    return found


def grab_window_rect(hwnd: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right, bottom


def img_to_jpeg_b64(img: Image.Image, quality: int = 70) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main() -> None:
    cfg = load_cfg()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if cfg.mqtt_user:
        client.username_pw_set(cfg.mqtt_user, cfg.mqtt_pass)
    client.connect(cfg.mqtt_host, cfg.mqtt_port, keepalive=30)

    pad = vg.VX360Gamepad()

    last_action: Dict[str, Any] = {"lx": 0, "ly": 0, "rx": 0, "ry": 0, "btn": {}}

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

    hwnd = None
    print(f"[win_agent] looking for process: {cfg.process_name}")
    for _ in range(50):
        hwnd = find_hwnd_by_process_name(cfg.process_name)
        if hwnd:
            break
        time.sleep(0.2)

    if not hwnd:
        raise RuntimeError(f"Не нашёл окно процесса {cfg.process_name}. Проверь GAME_PROCESS.")

    print(f"[win_agent] found hwnd={hwnd}")

    dt = 1.0 / max(cfg.capture_fps, 1.0)

    while True:
        t0 = time.time()

        l, t, r, b = grab_window_rect(hwnd)
        monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        shot = sct.grab(monitor)

        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img_small = img.resize((256, 256))
        jpeg_b64 = img_to_jpeg_b64(img_small)

        frame_msg = {
            "ts": time.time(),
            "process": cfg.process_name,
            "resize": [256, 256],
            "img_b64_jpg": jpeg_b64,
        }
        client.publish(cfg.topic_frame, json.dumps(frame_msg), qos=0)

        lx = clamp(float(last_action.get("lx", 0.0)), -1.0, 1.0)
        ly = clamp(float(last_action.get("ly", 0.0)), -1.0, 1.0)
        rx = clamp(float(last_action.get("rx", 0.0)), -1.0, 1.0)
        ry = clamp(float(last_action.get("ry", 0.0)), -1.0, 1.0)

        pad.left_joystick(int(lx * 32767), int(-ly * 32767))
        pad.right_joystick(int(rx * 32767), int(-ry * 32767))

        btn = last_action.get("btn", {}) or {}

        def set_btn(flag: bool, button):
            if flag:
                pad.press_button(button)
            else:
                pad.release_button(button)

        set_btn(bool(btn.get("A", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        set_btn(bool(btn.get("B", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        set_btn(bool(btn.get("X", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        set_btn(bool(btn.get("Y", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        set_btn(bool(btn.get("LB", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        set_btn(bool(btn.get("RB", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        set_btn(bool(btn.get("START", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        set_btn(bool(btn.get("BACK", 0)), vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)

        pad.update()

        spent = time.time() - t0
        if spent < dt:
            time.sleep(dt - spent)


if __name__ == "__main__":
    main()
