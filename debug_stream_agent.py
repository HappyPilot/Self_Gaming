#!/usr/bin/env python3
"""Visual Debugger: Streams video with overlay of AI perception and intent."""
import base64
import json
import logging
import os
import threading
import time
from collections import deque
from io import BytesIO

import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template_string
from PIL import Image, ImageDraw

from utils.frame_transport import get_frame_bytes

# --- Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
OBJECT_TOPIC = os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
CURSOR_TOPIC = os.getenv("CURSOR_TOPIC", "cursor/state")
HTTP_PORT = int(os.getenv("DEBUG_PORT", "5000"))
DECODE_LOG_INTERVAL_SEC = float(os.getenv("DEBUG_STREAM_DECODE_LOG_SEC", "5.0"))
DEBUG_PUBLISH_TOPIC = os.getenv("DEBUG_PUBLISH_TOPIC", "vision/frame/debug")
DEBUG_PUBLISH_ENABLED = os.getenv("DEBUG_PUBLISH_ENABLED", "0").lower() in ("1", "true", "yes", "on")
DEBUG_PUBLISH_FPS = float(os.getenv("DEBUG_PUBLISH_FPS", "1.0"))
DEBUG_PUBLISH_QUALITY = int(os.getenv("DEBUG_PUBLISH_QUALITY", "80"))
CURSOR_STALE_SEC = float(os.getenv("CURSOR_STALE_SEC", "2.0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_stream")

app = Flask(__name__)

# --- State ---
state_lock = threading.Lock()
latest_frame = None
latest_objects = []
latest_action = None
latest_scene_text = ""
latest_cursor = None
action_history = deque(maxlen=20)
last_decode_error_ts = 0.0


def should_publish_debug(last_ts: float, now: float, fps: float) -> bool:
    if fps <= 0:
        return False
    interval = 1.0 / fps
    return (now - last_ts) >= interval


def control_point_from_action(action, width: int, height: int):
    if not action:
        return None
    target = action.get("target_norm")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        try:
            x_norm = float(target[0])
            y_norm = float(target[1])
        except (TypeError, ValueError):
            return None
        return (int(round(x_norm * width)), int(round(y_norm * height)))
    if "dx" in action and "dy" in action:
        try:
            dx = float(action.get("dx", 0.0))
            dy = float(action.get("dy", 0.0))
        except (TypeError, ValueError):
            return None
        return (int(round(width / 2 + dx)), int(round(height / 2 + dy)))
    return None


def build_debug_payload(img: Image.Image, timestamp: float, source: str = "debug_stream", quality: int = 80) -> dict:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "ok": True,
        "timestamp": timestamp,
        "width": img.width,
        "height": img.height,
        "source": source,
        "image_b64": b64,
    }


def _cursor_point(cursor, width: int, height: int, now: float):
    if not cursor or not cursor.get("ok"):
        return None
    ts = cursor.get("timestamp") or cursor.get("ts") or 0.0
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        ts = 0.0
    if ts and (now - ts) > CURSOR_STALE_SEC:
        return None
    try:
        x_norm = float(cursor.get("x_norm"))
        y_norm = float(cursor.get("y_norm"))
    except (TypeError, ValueError):
        return None
    return (int(round(x_norm * width)), int(round(y_norm * height)))


def render_annotated_frame(
    frame: Image.Image,
    objects: list,
    action,
    scene_text: str,
    cursor,
) -> Image.Image:
    img = frame.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for obj in objects:
        bbox = obj.get("bbox") or obj.get("box")
        label = obj.get("label") or obj.get("class") or "?"
        if bbox:
            if bbox[0] <= 1.0:
                x = int(bbox[0] * w)
                y = int(bbox[1] * h)
                bw = int(bbox[2] * w)
                bh = int(bbox[3] * h)
            else:
                x, y, bw, bh = [int(v) for v in bbox]
            draw.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 0), width=2)
            draw.text((x, max(0, y - 12)), label, fill=(0, 255, 0))

    act = action.get("action", "") if action else ""
    if action and "move" in act:
        dx = int(action.get("dx", 0))
        dy = int(action.get("dy", 0))
        cx, cy = w // 2, h // 2
        draw.line([(cx, cy), (cx + dx, cy + dy)], fill=(255, 0, 0), width=2)
        end_x, end_y = cx + dx, cy + dy
        draw.ellipse([end_x - 4, end_y - 4, end_x + 4, end_y + 4], outline=(255, 0, 0), width=2)

    control_point = control_point_from_action(action, w, h)
    if control_point:
        cx, cy = control_point
        draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 0, 255), width=2)
        draw.line([(cx, cy - 6), (cx, cy + 6)], fill=(255, 0, 255), width=2)

    cursor_point = _cursor_point(cursor, w, h, time.time())
    if cursor_point:
        cx, cy = cursor_point
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(0, 128, 255), width=2)

    status = f"Act: {act} | Scene: {scene_text}"
    draw.text((10, max(0, h - 20)), status, fill=(255, 255, 0))
    return img

# --- MQTT Worker ---
def on_connect(client, userdata, flags, rc):
    client.subscribe([(FRAME_TOPIC, 0), (OBJECT_TOPIC, 0), (ACT_TOPIC, 0), (SCENE_TOPIC, 0), (CURSOR_TOPIC, 0)])
    logger.info("Connected to MQTT")

def on_message(client, userdata, msg):
    global latest_frame, latest_objects, latest_action, latest_scene_text, latest_cursor, last_decode_error_ts
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        
        if msg.topic == FRAME_TOPIC:
            data = get_frame_bytes(payload)
            if data:
                try:
                    img = Image.open(BytesIO(data)).convert("RGB")
                except Exception as exc:
                    now = time.time()
                    if now - last_decode_error_ts >= DECODE_LOG_INTERVAL_SEC:
                        logger.warning("Failed to decode frame: %s", exc)
                        last_decode_error_ts = now
                    return
                with state_lock:
                    latest_frame = img
        
        elif msg.topic == OBJECT_TOPIC:
            with state_lock:
                latest_objects = payload.get("objects", [])
        
        elif msg.topic == ACT_TOPIC:
            with state_lock:
                latest_action = payload
                action_history.append(payload.get("action", "?"))
        
        elif msg.topic == SCENE_TOPIC:
            text = payload.get("text", [])
            if isinstance(text, list):
                text = " ".join(text)
            with state_lock:
                latest_scene_text = text[:100]
        elif msg.topic == CURSOR_TOPIC:
            if isinstance(payload, dict):
                with state_lock:
                    latest_cursor = payload

    except Exception as e:
        logger.error(f"Error processing message: {e}")

def mqtt_loop():
    client = mqtt.Client(client_id="debug_stream", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_forever()

# --- Video Generator ---
def generate_frames():
    while True:
        with state_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            objects = list(latest_objects)
            action = dict(latest_action) if latest_action else None
            scene_text = latest_scene_text
            cursor = dict(latest_cursor) if latest_cursor else None
        if frame is None:
            time.sleep(0.1)
            continue

        img = render_annotated_frame(frame, objects, action, scene_text, cursor)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        frame_bytes = buffer.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) # Cap at ~20 FPS for viewing


def publish_loop():
    if not DEBUG_PUBLISH_ENABLED:
        logger.info("Debug publish disabled")
        return
    client = mqtt.Client(client_id="debug_stream_pub", protocol=mqtt.MQTTv311)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()
    last_pub = 0.0
    while True:
        with state_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            objects = list(latest_objects)
            action = dict(latest_action) if latest_action else None
            scene_text = latest_scene_text
            cursor = dict(latest_cursor) if latest_cursor else None
        if frame is None:
            time.sleep(0.1)
            continue
        now = time.time()
        if should_publish_debug(last_pub, now, DEBUG_PUBLISH_FPS):
            img = render_annotated_frame(frame, objects, action, scene_text, cursor)
            payload = build_debug_payload(img, now, quality=DEBUG_PUBLISH_QUALITY)
            client.publish(DEBUG_PUBLISH_TOPIC, json.dumps(payload))
            last_pub = now
        time.sleep(0.05)

@app.route('/')
def index():
    html = """
        <html>
        <head><title>Agent Vision Debug</title></head>
        <body style="background: #111; color: #eee; font-family: monospace; text-align: center;">
            <h1>Agent Vision Stream</h1>
            <img src="/video_feed" style="border: 2px solid #444; max-width: 90%; height: auto;">
            <div style="margin-top: 20px;">
                <h3>Active Agents</h3>
                <p>Vision -> Object Det -> Scene -> Policy -> Act</p>
            </div>
        </body>
        </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=mqtt_loop, daemon=True)
    t.start()
    if DEBUG_PUBLISH_ENABLED:
        pub_thread = threading.Thread(target=publish_loop, daemon=True)
        pub_thread.start()
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False)
