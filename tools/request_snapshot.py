import json
import time

import paho.mqtt.client as mqtt

result = {}

def on_snapshot(client, userdata, msg):
    result['payload'] = msg.payload.decode('utf-8')
    client.loop_stop()

cli = mqtt.Client()
cli.on_message = on_snapshot
cli.connect('127.0.0.1', 1883, 60)
cli.subscribe('vision/snapshot')
cli.loop_start()
cli.publish('vision/cmd', json.dumps({'cmd': 'snapshot'}))
for _ in range(50):
    if 'payload' in result:
        break
    time.sleep(0.1)
cli.loop_stop()
cli.disconnect()
print(result.get('payload', 'NO_SNAPSHOT'))
