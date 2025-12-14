import json
import sys
import time

import paho.mqtt.client as mqtt

timeout_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

response = {}


def on_message(client, userdata, msg):
    response['payload'] = msg.payload.decode('utf-8')
    client.loop_stop()


cli = mqtt.Client()
cli.on_message = on_message
cli.connect('127.0.0.1', 1883, 60)
cli.loop_start()
cli.subscribe('ocr_easy/text')
cli.publish('ocr_easy/cmd', json.dumps({'cmd': 'once', 'timeout': timeout_seconds}))
for _ in range(int(timeout_seconds * 10)):
    if 'payload' in response:
        break
    time.sleep(0.1)
cli.loop_stop()
cli.disconnect()
print(response.get('payload', 'NO_RESPONSE'))
