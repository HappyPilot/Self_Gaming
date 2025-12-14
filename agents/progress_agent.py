#!/usr/bin/env python3
"""
Progress Agent (Meta-Controller).
Monitors learning progress. If stagnation is detected, consults LLM and intervenes.
Exposes a Web Dashboard on port 8000.
"""
import json
import logging
import os
import threading
import time
import requests
from collections import deque
from flask import Flask, jsonify, render_template_string
import paho.mqtt.client as mqtt

# --- CONFIG ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
REWARD_TOPIC = os.getenv("REWARD_TOPIC", "train/reward")
SCENE_TOPIC = os.getenv("SCENE_TOPIC", "scene/state")
OBJECT_TOPIC = os.getenv("OBJECT_TOPIC", "vision/objects")
OCR_TOPIC = os.getenv("OCR_TOPIC", "ocr_easy/text")
SIMPLE_OCR_TOPIC = os.getenv("SIMPLE_OCR_TOPIC", "simple_ocr/text")
ACT_CMD_TOPIC = os.getenv("ACT_CMD_TOPIC", "act/cmd")
LLM_ENDPOINT = os.getenv("TEACHER_LOCAL_ENDPOINT", "http://10.0.0.230:11434/v1/chat/completions")
LLM_MODEL = os.getenv("TEACHER_OPENAI_MODEL", "gpt-4o-mini")

STUCK_THRESHOLD_SEC = 300  # 5 minutes without significant reward
MIN_REWARD_THRESHOLD = 0.05 # What counts as "good" reward
HTTP_PORT = 8000

logging.basicConfig(level=logging.INFO, format="[progress] %(message)s")
logger = logging.getLogger("progress")

# --- STATE ---
state = {
    "last_reward_time": time.time(),
    "total_reward": 0.0,
    "recent_rewards": deque(maxlen=100),
    "status": "OK",
    "last_intervention": "None",
    "scene_desc": "Unknown",
    "scene_ts": 0.0,
    "objects": {"count": 0, "labels": [], "ts": 0.0},
    "ocr": {"text": "", "ts": 0.0},
    "simple_ocr": {"text": "", "ts": 0.0},
}
lock = threading.Lock()

def _now() -> float:
    return time.time()

def _age(ts: float) -> int:
    if not ts:
        return -1
    return int(max(0, _now() - ts))

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    topics = [(REWARD_TOPIC, 0), (SCENE_TOPIC, 0), (OBJECT_TOPIC, 0), (OCR_TOPIC, 0), (SIMPLE_OCR_TOPIC, 0)]
    client.subscribe(topics)
    logger.info("Connected to MQTT")

def on_message(client, userdata, msg):
    global state
    try:
        payload = json.loads(msg.payload.decode())
        
        if msg.topic == REWARD_TOPIC:
            r = float(payload.get("reward", 0.0))
            with lock:
                if r > MIN_REWARD_THRESHOLD:
                    state["last_reward_time"] = _now()
                state["total_reward"] += r
                state["recent_rewards"].append(r)
                
        elif msg.topic == SCENE_TOPIC:
            txt = payload.get("text", [])
            with lock:
                state["scene_desc"] = " ".join(txt)[:200]
                state["scene_ts"] = payload.get("timestamp", _now())
        elif msg.topic == OBJECT_TOPIC:
            objects = payload.get("objects") or []
            labels = []
            for obj in objects:
                lbl = obj.get("label") or obj.get("class") or ""
                if lbl:
                    labels.append(str(lbl))
            with lock:
                state["objects"] = {
                    "count": len(objects),
                    "labels": labels[:8],
                    "ts": payload.get("timestamp", _now()),
                }
        elif msg.topic == OCR_TOPIC:
            txt = payload.get("text") if isinstance(payload, dict) else payload
            with lock:
                state["ocr"] = {"text": str(txt)[:200], "ts": _now()}
        elif msg.topic == SIMPLE_OCR_TOPIC:
            txt = payload.get("text") if isinstance(payload, dict) else payload
            with lock:
                state["simple_ocr"] = {"text": str(txt)[:200], "ts": _now()}
                
    except Exception as e:
        pass

mqtt_client = mqtt.Client(client_id="progress_agent", protocol=mqtt.MQTTv311)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# --- LLM INTERVENTION ---
def consult_oracle():
    """Asks LLM what to do when stuck."""
    try:
        with lock:
            scene = state["scene_desc"]
        
        prompt = f"""
        I am an AI playing a video game. I have been stuck for 5 minutes (no reward).
        Screen text: "{scene}".
        
        Suggest ONE key press to unstick me (e.g., Esc, Space, Enter, I, M).
        Return ONLY the key name.
        """
        
        logger.info("Consulting LLM...")
        # Support for Local LLM (Ollama style) or OpenAI
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False
        }
        
        # Simple request (assuming local endpoint doesn't need auth, or use env var)
        headers = {}
        if "api.openai.com" in LLM_ENDPOINT:
            headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

        resp = requests.post(LLM_ENDPOINT, json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            ans = resp.json()
            # Extract content (handle standard OpenAI format)
            if "choices" in ans:
                suggestion = ans["choices"][0]["message"]["content"].strip().split()[0] # Take first word
                return suggestion
        return None
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None

def perform_intervention(key):
    logger.warning(f"⚠️ INTERVENTION: Pressing '{key}'")
    payload = {
        "action": "keyboard",
        "key": key.lower(),
        "source": "progress_agent_intervention"
    }
    mqtt_client.publish(ACT_CMD_TOPIC, json.dumps(payload))
    with lock:
        state["last_intervention"] = f"Pressed {key} at {time.strftime('%H:%M:%S')}"
        state["last_reward_time"] = time.time() # Reset timer so we don't spam

def monitor_loop():
    while True:
        time.sleep(10)
        now = time.time()
        with lock:
            last = state["last_reward_time"]
            delta = now - last
        
        if delta > STUCK_THRESHOLD_SEC:
            with lock: state["status"] = "STUCK"
            key = consult_oracle()
            if key:
                perform_intervention(key)
        else:
            with lock: state["status"] = "OK"

# --- LOG READER ---
def read_thought_log(n=5):
    log_path = "/app/logs/thought_process.log"
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-n:]
                for line in reversed(lines):
                    try:
                        entries.append(json.loads(line))
                    except: pass
        except Exception: pass
    return entries

# --- WEB SERVER ---
app = Flask(__name__)

@app.route('/')
def dashboard():
    with lock:
        s = state.copy()
        last_reward_ago = int(time.time() - s["last_reward_time"])
        scene_age = _age(s.get("scene_ts", 0))
        obj_state = s.get("objects", {})
        obj_age = _age(obj_state.get("ts", 0))
        ocr_age = _age(s.get("ocr", {}).get("ts", 0))
        simple_age = _age(s.get("simple_ocr", {}).get("ts", 0))
    
    thoughts = read_thought_log()
    color = "green" if s["status"] == "OK" else "red"
    
    return render_template_string("""
    <html>
    <head>
        <meta refresh="10">
        <style>
            body { background: #111; color: #eee; font-family: monospace; padding: 20px; }
            .box { border: 1px solid #444; padding: 20px; margin: 10px; border-radius: 5px; background: #222; }
            .status { font-size: 24px; font-weight: bold; color: {{ color }}; }
            .thought { margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px; }
            .advice { color: #8f8; font-weight: bold; }
            .scene { color: #aaa; font-size: 0.9em; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
            .label { color: #888; }
        </style>
    </head>
    <body>
        <h1>Progress Manager</h1>
        <div class="grid">
            <div class="box">
                Status: <span class="status">{{ s.status }}</span><br>
                Last Reward: {{ last_reward_ago }}s ago<br>
                Total Reward: {{ s.total_reward | round(2) }}<br>
                Intervention: {{ s.last_intervention }}
            </div>
            <div class="box">
                <div class="label">Scene text (age {{ scene_age }}s)</div>
                <div>{{ s.scene_desc }}</div>
            </div>
            <div class="box">
                <div class="label">Objects (age {{ obj_age }}s)</div>
                <div>Count: {{ obj_state.count or 0 }}</div>
                <div>Labels: {{ obj_state.labels }}</div>
            </div>
            <div class="box">
                <div class="label">OCR Easy (age {{ ocr_age }}s)</div>
                <div>{{ s.ocr.text if s.ocr else "" }}</div>
            </div>
            <div class="box">
                <div class="label">Simple OCR (age {{ simple_age }}s)</div>
                <div>{{ s.simple_ocr.text if s.simple_ocr else "" }}</div>
            </div>
        </div>
        
        <div class="box">
            <h3>Teacher's Thoughts (Latest)</h3>
            {% for t in thoughts %}
            <div class="thought">
                <div class="advice">💡 {{ t.advice }}</div>
                <div class="scene">👁️ {{ t.scene }}</div>
                <div style="font-size:0.8em; color:#666;">{{ t.timestamp }} | Game: {{ t.game }}</div>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """, s=s, last_reward_ago=last_reward_ago, color=color, thoughts=thoughts,
    scene_age=scene_age, obj_state=obj_state, obj_age=obj_age, ocr_age=ocr_age, simple_age=simple_age)

@app.route('/api/status')
def api_status():
    with lock:
        status = state.copy()
        status["recent_rewards"] = list(state.get("recent_rewards", []))
        status["age"] = {
            "reward": _age(state.get("last_reward_time", 0)),
            "scene": _age(state.get("scene_ts", 0)),
            "objects": _age(state.get("objects", {}).get("ts", 0)),
            "ocr": _age(state.get("ocr", {}).get("ts", 0)),
            "simple_ocr": _age(state.get("simple_ocr", {}).get("ts", 0)),
        }
        return jsonify(status)

def main():
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
    mqtt_client.loop_start()
    
    t = threading.Thread(target=monitor_loop, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=HTTP_PORT)

if __name__ == "__main__":
    main()
