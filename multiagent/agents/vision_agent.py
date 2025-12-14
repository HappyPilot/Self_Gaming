#!/usr/bin/env python3
import os
import json
import base64
import time
import signal
import threading

import cv2
import numpy as np
import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("MQTT_HOST", "mq")  # имя контейнера брокера в docker-compose сети
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_CMD = os.environ.get("MQTT_TOPIC_CMD", "vision/cmd")
MQTT_TOPIC_METRIC = os.environ.get("MQTT_TOPIC_METRIC", "vision/mean")
MQTT_TOPIC_SNAPSHOT = os.environ.get("MQTT_TOPIC_SNAPSHOT", "vision/snapshot")
MQTT_TOPIC_FRAME = os.environ.get("VISION_FRAME_TOPIC", "vision/frame")
FRAME_PUBLISH_INTERVAL = float(os.environ.get("VISION_FRAME_INTERVAL", "0.5"))
FRAME_JPEG_QUALITY = int(os.environ.get("VISION_FRAME_JPEG_QUALITY", "70"))
FRAME_STREAM_ENABLED = os.environ.get("VISION_FRAME_ENABLED", "1") not in {"0", "false", "False"}
VISION_CONFIG_TOPIC = os.environ.get("VISION_CONFIG_TOPIC", "vision/config")
VISION_STATUS_TOPIC = os.environ.get("VISION_STATUS_TOPIC", "vision/status")
VISION_MODE_DEFAULT = os.environ.get("VISION_MODE_DEFAULT", "medium").lower()
VISION_STATUS_INTERVAL = float(os.environ.get("VISION_STATUS_INTERVAL", "15"))


def _float_env(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


MODE_FRAME_INTERVALS = {
    "low": _float_env("VISION_FRAME_INTERVAL_LOW", FRAME_PUBLISH_INTERVAL * 1.5 or 0.8),
    "medium": _float_env("VISION_FRAME_INTERVAL_MED", FRAME_PUBLISH_INTERVAL),
    "high": _float_env("VISION_FRAME_INTERVAL_HIGH", max(0.1, FRAME_PUBLISH_INTERVAL * 0.5)),
}

config_lock = threading.Lock()
current_mode = VISION_MODE_DEFAULT if VISION_MODE_DEFAULT in MODE_FRAME_INTERVALS else "medium"
current_frame_interval = MODE_FRAME_INTERVALS.get(current_mode, MODE_FRAME_INTERVALS["medium"])
last_config_reason = "default"


def publish_status(client: mqtt.Client):
    if not VISION_STATUS_TOPIC:
        return
    with config_lock:
        payload = {
            "ok": True,
            "event": "vision_status",
            "mode": current_mode,
            "frame_interval": round(current_frame_interval, 3),
            "reason": last_config_reason,
        }
    try:
        client.publish(VISION_STATUS_TOPIC, json.dumps(payload), qos=0, retain=False)
    except Exception:
        pass


def apply_vision_mode(config: dict, client: mqtt.Client | None = None):
    """Update mode/interval from config payload."""

    mode = str(config.get("mode") or config.get("vision_mode") or "").lower()
    if mode not in MODE_FRAME_INTERVALS:
        return
    reason = config.get("reason", "config")
    with config_lock:
        global current_mode, current_frame_interval, last_config_reason
        current_mode = mode
        current_frame_interval = MODE_FRAME_INTERVALS[mode]
        last_config_reason = reason
    if client:
        publish_status(client)


def get_frame_interval() -> float:
    with config_lock:
        return current_frame_interval

# Источник видео и желаемые параметры (можно переопределять через ENV)
VIDEO_DEVICE = os.environ.get("VIDEO_DEVICE", "/dev/video0")
VIDEO_FALLBACKS = [p.strip() for p in os.environ.get("VIDEO_DEVICE_FALLBACKS", "/dev/video2,/dev/video1,0,2").split(",") if p.strip()]
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "0"))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "0"))
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "0"))
VIDEO_PIXFMT = os.environ.get("VIDEO_PIXFMT", "")[:4]

stop_event = threading.Event()

# --------- MQTT callbacks (API v2) ----------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        topics = [(MQTT_TOPIC_CMD, 0)]
        if VISION_CONFIG_TOPIC:
            topics.append((VISION_CONFIG_TOPIC, 0))
        client.subscribe(topics)
        print(
            json.dumps(
                {
                    "ok": True,
                    "event": "connected",
                    "subscribed": [t[0] for t in topics],
                }
            ),
            flush=True,
        )
    else:
        print(json.dumps({"ok": False, "event": "connect_failed", "code": int(reason_code)}), flush=True)

def on_message(client, userdata, msg):
    raw = msg.payload.decode("utf-8", "ignore")
    if msg.topic == VISION_CONFIG_TOPIC:
        try:
            data = json.loads(raw)
        except Exception:
            return
        apply_vision_mode(data, client)
        return
    try:
        payload = raw.strip()
        data = json.loads(payload) if payload.startswith("{") else {"cmd": payload}
    except Exception:
        data = {"cmd": raw}

    cmd = (data.get("cmd") or data.get("action") or "").lower()

    if cmd == "ping":
        client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": True, "pong": True}))
    elif cmd == "snapshot":
        frame = userdata.get("last_frame")
        if frame is None:
            client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": False, "error": "no_frame"}))
            return
        # Кодируем JPEG и шлём base64
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": False, "error": "encode_fail"}))
            return
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
        client.publish(MQTT_TOPIC_SNAPSHOT, json.dumps({"ok": True, "image_b64": b64}))
    elif cmd in ("stop", "quit", "exit"):
        stop_event.set()
    else:
        client.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": False, "error": "unknown_cmd", "cmd": cmd}))

def on_disconnect(client, userdata, reason_code, properties=None):
    print(json.dumps({"ok": False, "event": "disconnected", "code": int(reason_code)}), flush=True)

# --------- Видео-цикл ----------
def _configure_capture(cap):
    # Настроить запрошенные параметры; игнорируем не поддерживаемые устройства
    if VIDEO_WIDTH > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    if VIDEO_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    if VIDEO_FPS > 0:
        cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
    if VIDEO_PIXFMT:
        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_PIXFMT)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass


def open_capture():
    # Перебираем кандидатов, начиная с основного устройства
    candidates = []
    seen = set()
    for cand in [VIDEO_DEVICE] + VIDEO_FALLBACKS:
        if not cand:
            continue
        # Числовые индексы передаем как int, остальные — строки
        key = cand
        if cand.isdigit():
            cand_obj = int(cand)
        else:
            cand_obj = cand
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand_obj)

    for cand in candidates:
        cap = cv2.VideoCapture(cand, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        _configure_capture(cap)
        return cap, cand

    return None, None

def compute_mean(frame):
    # Универсально считаем среднюю яркость: если 3 канала — в GRAY, если 1 — сразу
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        gray = frame
    else:
        # неожиданный формат (например, 4 канала) — конвертнём в BGR, потом в GRAY
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def run():
    userdata = {"last_frame": None, "last_frame_pub": 0.0}

    # Новый клиент без DeprecationWarning:
    # указываем callback_api_version и протокол явно
    cli = mqtt.Client(
        client_id="vision",
        userdata=userdata,
        protocol=mqtt.MQTTv311,
        transport="tcp",
    )
    cli.on_connect = on_connect
    cli.on_message = on_message
    cli.on_disconnect = on_disconnect

    # Подключаемся (блокировка не нужна — будет loop_start)
    cli.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    cli.loop_start()

    cap, active_device = open_capture()
    if cap is None or not cap.isOpened():
        print(json.dumps({"ok": False, "error": "capture_open_failed", "device": VIDEO_DEVICE, "fallbacks": VIDEO_FALLBACKS}), flush=True)
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = round(cap.get(cv2.CAP_PROP_FPS) or 0, 2)
    print(
        json.dumps(
            {
                "ok": True,
                "event": "capture_opened",
                "device": str(active_device),
                "width": actual_width,
                "height": actual_height,
                "fps": actual_fps,
            }
        ),
        flush=True,
    )

    last_status = 0.0
    publish_status(cli)
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            userdata["last_frame"] = frame

            m = compute_mean(frame)
            if m is not None:
                # Периодически публикуем метрику яркости
                cli.publish(MQTT_TOPIC_METRIC, json.dumps({"ok": True, "mean": round(m, 2)}), qos=0, retain=False)

            if FRAME_STREAM_ENABLED:
                now = time.time()
                frame_interval = get_frame_interval()
                if now - userdata["last_frame_pub"] >= frame_interval:
                    userdata["last_frame_pub"] = now
                    ok, jpg = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(FRAME_JPEG_QUALITY, 95))],
                    )
                    if ok:
                        frame_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
                        cli.publish(
                            MQTT_TOPIC_FRAME,
                            json.dumps(
                                {
                                    "ok": True,
                                    "timestamp": now,
                                    "image_b64": frame_b64,
                                    "width": frame.shape[1],
                                    "height": frame.shape[0],
                                }
                            ),
                            qos=0,
                            retain=False,
                        )

            if VISION_STATUS_TOPIC and time.time() - last_status >= VISION_STATUS_INTERVAL:
                publish_status(cli)
                last_status = time.time()

            # небольшая пауза, чтобы не заваливать брокера
            time.sleep(0.05)

    finally:
        cap.release()
        cli.loop_stop()
        cli.disconnect()
        print(json.dumps({"ok": True, "event": "stopped"}), flush=True)

def _sigterm(*_):
    stop_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)
    run()
