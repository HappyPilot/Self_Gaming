import json
import sys

import paho.mqtt.client as mqtt

topic = sys.argv[1]
payload = json.loads(sys.argv[2])
cli = mqtt.Client()
cli.connect('127.0.0.1', 1883, 60)
cli.publish(topic, json.dumps(payload))
cli.disconnect()
