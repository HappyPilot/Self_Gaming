#!/usr/bin/env bash
python3 - <<'PY'
import json
import sys
try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit('paho-mqtt not installed inside Jetson env')
client = mqtt.Client()
client.connect('127.0.0.1', 1883, 60)
client.loop_start()
client.publish('teach/cmd', json.dumps({'cmd': 'plan'}))
client.loop_stop()
client.disconnect()
PY
