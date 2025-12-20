import argparse
import base64
import json
import time
from typing import List

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from ultralytics import YOLO


def decode_frame(payload: bytes):
    try:
        msg = json.loads(payload.decode("utf-8", "ignore"))
    except Exception:
        return None
    encoded = msg.get("image_b64")
    if not encoded:
        return None
    try:
        data = base64.b64decode(encoded)
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--frame-topic", default="vision/frame/preview")
    parser.add_argument("--topic", default="vision/observation")
    parser.add_argument("--model", default="/mnt/ssd/models/yolo/yolov8s-world.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[
            "enemy",
            "monster",
            "player",
            "portal",
            "waypoint",
            "chest",
            "loot",
            "gold",
            "npc",
            "boss",
            "obstacle",
            "projectile",
        ],
    )
    args = parser.parse_args()

    print(f"[yolo_world_from_mqtt] frame topic: {args.frame_topic}", flush=True)

    model = YOLO(args.model)
    device = "cpu"  # force CPU to avoid GPU alloc issues on Jetson
    model.to(device)

    client = mqtt.Client(client_id="yolo_world_mqtt_frames", protocol=mqtt.MQTTv311)
    client.connect(args.mqtt_host, args.mqtt_port, 30)
    client.loop_start()

    def on_message(_cli, _ud, msg):
        img = decode_frame(msg.payload)
        if img is None:
            return
        res = model.predict(
            img,
            device=device,
            verbose=False,
            conf=args.conf,
            classes=None,
            stream=False,
            save=False,
            imgsz=640,
            half=False,
            augment=False,
            max_det=200,
        )
        if not res:
            return
        dets = []
        for b in res[0].boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            if score < args.conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append(
                {
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                    "class_id": cls_id,
                    "class_name": args.classes[cls_id] if cls_id < len(args.classes) else str(cls_id),
                }
            )
        payload = {"timestamp": time.time(), "source": f"yolo_world_{device}", "detections": dets}
        client.publish(args.topic, json.dumps(payload))

    client.subscribe(args.frame_topic, qos=0)
    client.on_message = on_message

    try:
        while True:
            time.sleep(1.0)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
