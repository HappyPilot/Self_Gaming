from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
import paho.mqtt.client as mqtt

from nitrogen_client import NitroGenClient, NitroGenConfig


@dataclass(frozen=True)
class Cfg:
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str
    mqtt_pass: str

    topic_frame: str
    topic_intent: str
    topic_action: str
    topic_telem: str

    nitrogen_host: str
    nitrogen_port: int
    nitrogen_timeout_s: float
    action_rate_hz: float


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

BUTTON_TOKENS = [
    "BACK",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_UP",
    "EAST",
    "GUIDE",
    "LEFT_SHOULDER",
    "LEFT_THUMB",
    "LEFT_TRIGGER",
    "NORTH",
    "RIGHT_BOTTOM",
    "RIGHT_LEFT",
    "RIGHT_RIGHT",
    "RIGHT_SHOULDER",
    "RIGHT_THUMB",
    "RIGHT_TRIGGER",
    "RIGHT_UP",
    "SOUTH",
    "START",
    "WEST",
]

BUTTON_ALIAS = {
    "A": "SOUTH",
    "B": "EAST",
    "X": "WEST",
    "Y": "NORTH",
    "LB": "LEFT_SHOULDER",
    "RB": "RIGHT_SHOULDER",
    "START": "START",
    "BACK": "BACK",
}


def _parse_host_port(value: str) -> tuple[str, int] | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"tcp://{raw}"
    parsed = urlparse(raw)
    if not parsed.hostname:
        return None
    return parsed.hostname, int(parsed.port or 5555)


def _load_nitrogen_host_port() -> tuple[str, int]:
    host = _env("NITROGEN_HOST", "").strip()
    port_raw = _env("NITROGEN_PORT", "").strip()
    parsed = _parse_host_port(_env("NITROGEN_BASE_URL", ""))

    if not host:
        host = parsed[0] if parsed else "localhost"

    if port_raw:
        port = int(port_raw)
    else:
        port = parsed[1] if parsed else 5555

    return host, port


def load_cfg() -> Cfg:
    host, port = _load_nitrogen_host_port()
    return Cfg(
        mqtt_host=_env("MQTT_HOST", "localhost"),
        mqtt_port=int(_env("MQTT_PORT", "1883")),
        mqtt_user=_env("MQTT_USER", ""),
        mqtt_pass=_env("MQTT_PASS", ""),
        topic_frame=_env("TOPIC_FRAME", "self_gaming/vision/frame"),
        topic_intent=_env("TOPIC_INTENT", "self_gaming/control/intent"),
        topic_action=_env("TOPIC_ACTION", "self_gaming/control/action"),
        topic_telem=_env("TOPIC_TELEM", "self_gaming/telemetry/nitrogen"),
        nitrogen_host=host,
        nitrogen_port=port,
        nitrogen_timeout_s=float(_env("NITROGEN_TIMEOUT_S", "5.0")),
        action_rate_hz=float(_env("ACTION_RATE_HZ", "5")),
    )


class State:
    last_intent: Dict[str, Any] = {}
    last_action_ts: float = 0.0


def normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize NitroGen server response into a stable gamepad action format."""
    if isinstance(raw, dict) and {"j_left", "j_right", "buttons"}.issubset(raw.keys()):
        def _last_vec(value: Any) -> np.ndarray:
            if value is None:
                return np.array([])
            arr = np.asarray(value)
            if arr.ndim >= 2:
                arr = arr[-1]
            if arr.ndim == 0:
                return np.array([])
            return arr.reshape(-1)

        j_left = _last_vec(raw.get("j_left"))
        j_right = _last_vec(raw.get("j_right"))
        buttons = _last_vec(raw.get("buttons"))

        def _axis(arr: np.ndarray, idx: int) -> float:
            if arr.size <= idx:
                return 0.0
            try:
                return float(arr[idx])
            except Exception:
                return 0.0

        btn: Dict[str, int] = {}
        for i, token in enumerate(BUTTON_TOKENS):
            if i >= buttons.size:
                break
            pressed = int(float(buttons[i]) > 0.5)
            btn[token] = pressed
        for alias, token in BUTTON_ALIAS.items():
            btn[alias] = int(btn.get(token, 0))

        return {
            "lx": _axis(j_left, 0),
            "ly": _axis(j_left, 1),
            "rx": _axis(j_right, 0),
            "ry": _axis(j_right, 1),
            "btn": btn,
            "raw": raw,
        }

    seq = raw.get("actions") if isinstance(raw, dict) else raw
    seq = raw.get("action") if isinstance(raw, dict) and seq is None else seq
    if isinstance(seq, dict) and "timesteps" in seq:
        seq = seq["timesteps"]
    step = seq[-1] if isinstance(seq, list) and seq else seq

    def f(i: int, default: float = 0.0) -> float:
        if isinstance(step, list) and len(step) > i:
            try:
                return float(step[i])
            except Exception:
                return default
        return default

    def b(i: int, default: int = 0) -> int:
        if isinstance(step, list) and len(step) > i:
            try:
                return int(step[i])
            except Exception:
                return default
        return default

    return {
        "lx": f(0),
        "ly": f(1),
        "rx": f(2),
        "ry": f(3),
        "btn": {
            "A": b(4),
            "B": b(5),
            "X": b(6),
            "Y": b(7),
            "LB": b(8),
            "RB": b(9),
            "START": b(10),
            "BACK": b(11),
        },
        "raw": raw,
    }


def main() -> None:
    cfg = load_cfg()
    st = State()

    ng = NitroGenClient(NitroGenConfig(cfg.nitrogen_host, cfg.nitrogen_port, cfg.nitrogen_timeout_s))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if cfg.mqtt_user:
        client.username_pw_set(cfg.mqtt_user, cfg.mqtt_pass)

    def publish_telem(kind: str, data: Dict[str, Any]) -> None:
        payload = {"ts": time.time(), "kind": kind, **data}
        client.publish(cfg.topic_telem, json.dumps(payload), qos=0)

    def on_connect(_c, _u, _f, rc, _p=None):
        if rc != 0:
            raise RuntimeError(f"MQTT connect failed rc={rc}")
        client.subscribe([(cfg.topic_frame, 0), (cfg.topic_intent, 0)])
        publish_telem("status", {"msg": "nitrogen-proxy connected"})

    def on_message(_c, _u, msg):
        try:
            if msg.topic == cfg.topic_intent:
                st.last_intent = json.loads(msg.payload.decode("utf-8"))
                return

            if msg.topic != cfg.topic_frame:
                return

            now = time.time()
            min_dt = 1.0 / max(cfg.action_rate_hz, 1.0)
            if now - st.last_action_ts < min_dt:
                return

            frame = json.loads(msg.payload.decode("utf-8"))
            jpeg_b64 = frame["img_b64_jpg"]

            t0 = time.time()
            raw = ng.infer(jpeg_b64, intent=st.last_intent)
            t1 = time.time()

            action = normalize_action(raw)
            out = {"ts": time.time(), **action, "lat_ms": int((t1 - t0) * 1000)}
            client.publish(cfg.topic_action, json.dumps(out), qos=0)

            st.last_action_ts = now
            publish_telem("infer", {"lat_ms": out["lat_ms"]})

        except Exception as e:
            publish_telem("error", {"err": str(e)})

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg.mqtt_host, cfg.mqtt_port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
