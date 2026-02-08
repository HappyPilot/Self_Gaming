#!/usr/bin/env python3
"""UI Region Agent

Builds stable text ROI regions from Paddle OCR boxes and publishes them to ocr/regions.
Algorithm: sliding window of OCR frames, merge boxes by IoU or center distance, compute stability.
"""
from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

import paho.mqtt.client as mqtt

# Environment / defaults
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OCR_EASY_TOPIC = os.getenv("OCR_EASY_TOPIC", "ocr_easy/text")
OCR_REGIONS_TOPIC = os.getenv("OCR_REGIONS_TOPIC", "ocr/regions")
REGION_WINDOW = int(os.getenv("REGION_WINDOW", "20"))
REGION_MIN_CONF = float(os.getenv("REGION_MIN_CONF", "0.6"))
REGION_STABILITY_MIN = float(os.getenv("REGION_STABILITY_MIN", "0.5"))
REGION_TOPK = int(os.getenv("REGION_TOPK", "6"))
REGION_IOU_JOIN = float(os.getenv("REGION_IOU_JOIN", "0.2"))
REGION_DIST_JOIN = float(os.getenv("REGION_DIST_JOIN", "0.05"))
REGION_PUBLISH_INTERVAL = float(os.getenv("REGION_PUBLISH_INTERVAL", "1.0"))

stop_event = threading.Event()


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def iou(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def center_dist(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    return math.hypot(acx - bcx, acy - bcy)


class UiRegionAgent:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="ui_region")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.window: Deque[List[Tuple[List[float], float]]] = deque(maxlen=REGION_WINDOW)
        self.lock = threading.Lock()
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)

    def _on_connect(self, cli, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            cli.subscribe(OCR_EASY_TOPIC)
        else:
            print(f"[ui_region] MQTT connect failed rc={rc}")

    def _on_message(self, _cli, _ud, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
            results = data.get("results") or []
            filtered = []
            for r in results:
                conf = float(r.get("conf", 0.0))
                if conf < REGION_MIN_CONF:
                    continue
                box = r.get("box") or r.get("bbox")
                if not box or len(box) != 4:
                    continue
                x, y, w, h = [clamp01(float(v)) for v in box]
                filtered.append(([x, y, w, h], conf))
            with self.lock:
                self.window.append(filtered)
        except Exception as exc:
            print(f"[ui_region] parse error: {exc}")

    def _publish_loop(self):
        while not stop_event.is_set():
            stop_event.wait(REGION_PUBLISH_INTERVAL)
            if stop_event.is_set():
                break
            with self.lock:
                window_copy = list(self.window)
            if not window_copy:
                payload = {"ts": time.time(), "regions": []}
                self.client.publish(OCR_REGIONS_TOPIC, json.dumps(payload))
                continue

            # Flatten with frame indices
            flat: List[Tuple[List[float], float, int]] = []
            for idx, entries in enumerate(window_copy):
                for box, conf in entries:
                    flat.append((box, conf, idx))

            # Attach frame_idx attribute to allow cluster frame tracking
            items_for_cluster = []
            for box, conf, fidx in flat:
                items_for_cluster.append((box, conf, fidx))

            clusters: List[Dict] = []
            for box, conf, fidx in items_for_cluster:
                matched = False
                for c in clusters:
                    if iou(box, c["box"]) >= REGION_IOU_JOIN or center_dist(box, c["box"]) < REGION_DIST_JOIN:
                        cx, cy, cw, ch = c["box"]
                        nx1, ny1 = min(cx, box[0]), min(cy, box[1])
                        nx2, ny2 = max(cx + cw, box[0] + box[2]), max(cy + ch, box[1] + box[3])
                        c["box"] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                        c["frames"].add(fidx)
                        c["conf_sum"] += conf
                        c["count"] += 1
                        matched = True
                        break
                if not matched:
                    clusters.append(
                        {
                            "box": box,
                            "frames": set([fidx]),
                            "conf_sum": conf,
                            "count": 1,
                        }
                    )

            total_frames = max(1, len(window_copy))
            regions = []
            for idx, c in enumerate(clusters):
                stability = len(c["frames"]) / float(total_frames)
                if stability < REGION_STABILITY_MIN:
                    continue
                bx = [clamp01(v) for v in c["box"]]
                regions.append(
                    {
                        "id": f"r{idx}",
                        "kind": "text_cluster",
                        "box": bx,
                        "stability": round(stability, 3),
                        "avg_conf": round(c["conf_sum"] / max(1, c["count"]), 3),
                        "source": "paddle_clusters",
                    }
                )
            regions.sort(key=lambda r: (r["stability"], r["avg_conf"]), reverse=True)
            regions = regions[:REGION_TOPK]
            payload = {"ts": time.time(), "regions": regions}
            self.client.publish(OCR_REGIONS_TOPIC, json.dumps(payload))

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.publish_thread.start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()


def _handle_signal(signum, _frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = UiRegionAgent()
    agent.run()


if __name__ == "__main__":
    main()
