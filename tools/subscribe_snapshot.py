import time
import paho.mqtt.client as mqtt

messages = []

def on_message(client, userdata, msg):
    messages.append(msg.payload.decode('utf-8'))
    client.loop_stop()

cli = mqtt.Client()
cli.on_message = on_message
cli.connect('127.0.0.1', 1883, 60)
cli.subscribe('vision/snapshot')
cli.loop_start()
for _ in range(50):
    if messages:
        break
    time.sleep(0.1)
cli.loop_stop()
cli.disconnect()
print(messages[0] if messages else 'NO_SNAPSHOT')
