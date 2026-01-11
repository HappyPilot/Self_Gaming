#!/usr/bin/env python3
"""Visual Debugger: Streams video with overlay of AI perception and intent."""
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
HTTP_PORT = int(os.getenv("DEBUG_PORT", "5000"))
DECODE_LOG_INTERVAL_SEC = float(os.getenv("DEBUG_STREAM_DECODE_LOG_SEC", "5.0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_stream")

app = Flask(__name__)

# --- State ---
state_lock = threading.Lock()
latest_frame = None
latest_objects = []
latest_action = None
latest_scene_text = ""
action_history = deque(maxlen=20)
last_decode_error_ts = 0.0

# --- MQTT Worker ---
def on_connect(client, userdata, flags, rc):
    client.subscribe([(FRAME_TOPIC, 0), (OBJECT_TOPIC, 0), (ACT_TOPIC, 0), (SCENE_TOPIC, 0)])
    logger.info("Connected to MQTT")

def on_message(client, userdata, msg):
    global latest_frame, latest_objects, latest_action, latest_scene_text, last_decode_error_ts
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
            if latest_frame is None:
                time.sleep(0.1)
                continue
            
            # Copy frame to draw on it
            img = latest_frame.copy()
            w, h = img.size
            draw = ImageDraw.Draw(img)
            
            # Draw Objects
            for obj in latest_objects:
                bbox = obj.get("bbox") or obj.get("box")
                label = obj.get("label") or obj.get("class") or "?"
                if bbox:
                    # Normalize if floats < 1.0
                    if bbox[0] <= 1.0:
                        x = int(bbox[0] * w)
                        y = int(bbox[1] * h)
                        bw = int(bbox[2] * w)
                        bh = int(bbox[3] * h)
                    else:
                        x, y, bw, bh = [int(v) for v in bbox]
                    
                    draw.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 0), width=2)
                    draw.text((x, max(0, y - 12)), label, fill=(0, 255, 0))

            # Draw Action Intent
            if latest_action:
                act = latest_action.get("action", "")
                if "move" in act:
                    dx = int(latest_action.get("dx", 0))
                    dy = int(latest_action.get("dy", 0))
                    cx, cy = w // 2, h // 2
                    draw.line([(cx, cy), (cx + dx, cy + dy)], fill=(255, 0, 0), width=2)
                    end_x, end_y = cx + dx, cy + dy
                    draw.ellipse([end_x - 4, end_y - 4, end_x + 4, end_y + 4], outline=(255, 0, 0), width=2)
                
                # Status Text Overlay
                status = f"Act: {act} | Scene: {latest_scene_text}"
                draw.text((10, max(0, h - 20)), status, fill=(255, 255, 0))

            # Encode back to JPG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            frame_bytes = buffer.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) # Cap at ~20 FPS for viewing

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
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False)
