#!/usr/bin/env python3
"""Visual Debugger: Streams video with overlay of AI perception and intent."""
import base64
import json
import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template_string

# --- Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VISION_FRAME_TOPIC", "vision/frame")
OBJECT_TOPIC = os.getenv("VISION_OBJECT_TOPIC", "vision/objects")
ACT_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
HTTP_PORT = int(os.getenv("DEBUG_PORT", "5000"))

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

# --- MQTT Worker ---
def on_connect(client, userdata, flags, rc):
    client.subscribe([(FRAME_TOPIC, 0), (OBJECT_TOPIC, 0), (ACT_TOPIC, 0), (SCENE_TOPIC, 0)])
    logger.info("Connected to MQTT")

def on_message(client, userdata, msg):
    global latest_frame, latest_objects, latest_action, latest_scene_text
    try:
        payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        
        if msg.topic == FRAME_TOPIC:
            b64 = payload.get("image_b64")
            if b64:
                # Decode JPEG
                data = base64.b64decode(b64)
                np_arr = np.frombuffer(data, np.uint8)
                with state_lock:
                    latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
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
            h, w = img.shape[:2]
            
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
                    
                    cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw Action Intent
            if latest_action:
                act = latest_action.get("action", "")
                if "move" in act:
                    dx = int(latest_action.get("dx", 0))
                    dy = int(latest_action.get("dy", 0))
                    cx, cy = w // 2, h // 2
                    cv2.arrowedLine(img, (cx, cy), (cx+dx, cy+dy), (0, 0, 255), 3)
                
                # Status Text Overlay
                status = f"Act: {act} | Scene: {latest_scene_text}"
                cv2.putText(img, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Encode back to JPG
            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()

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
