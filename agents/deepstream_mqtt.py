#!/usr/bin/env python3
"""
DeepStream pipeline:
v4l2src -> nvvidconv -> nvstreammux -> nvinfer (YOLO11 TRT engine) -> appsink probe -> MQTT

We rely on raw tensor meta from nvinfer (output-tensor-meta=1) and parse output0 manually.
Assumptions:
- Engine input: 1x3x416x416, tensor name "images"
- Output: (1, 84, N) where first 4 are xywh, next 80 are class logits/probs.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from typing import List

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject  # type: ignore
import numpy as np
import paho.mqtt.client as mqtt

try:
    import pyds  # type: ignore
except ImportError as exc:  # pragma: no cover
    print("pyds not found; DeepStream bindings are required inside this container.", file=sys.stderr)
    raise


MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OBS_TOPIC = os.getenv("PERCEPTION_TOPIC", "vision/observation")
V4L2_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video0")
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1280"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "720"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")  # if set to http/rtsp, use that instead of v4l2
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.6"))
ENGINE_CONFIG = os.getenv("ENGINE_CONFIG", "/app/docker/deepstream/yolo11_pgie.txt")


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xc, yc, w, h = xywh
    return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dtype=np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (
            (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            - inter
            + 1e-6
        )
        idxs = rest[iou < iou_thresh]
    return keep


def parse_yolo_output(layer: pyds.NvDsInferLayerInfo, conf_thresh: float, iou_thresh: float) -> List[dict]:
    dims = layer.inferDims
    # Expected (1, 84, N)
    size = dims.d[0] * dims.d[1] * dims.d[2]
    ptr = ctypes.cast(layer.buffer, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(ptr, shape=(size,))
    data = data.reshape((dims.d[0], dims.d[1], dims.d[2]))

    boxes_xyxy = []
    scores = []
    classes = []
    for i in range(dims.d[2]):
        xywh = data[0, 0:4, i]
        cls_logits = data[0, 4:, i]
        cls_scores = 1 / (1 + np.exp(-cls_logits))  # sigmoid
        cls_id = int(cls_scores.argmax())
        score = float(cls_scores[cls_id])
        if score < conf_thresh:
            continue
        boxes_xyxy.append(xywh_to_xyxy(xywh))
        scores.append(score)
        classes.append(cls_id)

    if not boxes_xyxy:
        return []
    boxes_xyxy = np.stack(boxes_xyxy, axis=0)
    scores_np = np.array(scores)
    keep = nms(boxes_xyxy, scores_np, iou_thresh)
    results = []
    for k in keep:
        x1, y1, x2, y2 = boxes_xyxy[k]
        results.append(
            {
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[k]),
                "class_id": int(classes[k]),
            }
        )
    return results


def infer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK
    print("[deepstream] frame received at pgie src")
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    detections: List[dict] = []
    l_user = batch_meta.batch_user_meta_list
    while l_user:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tmeta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            for i in range(tmeta.num_output_layers):
                layer = pyds.get_nvds_LayerInfo(tmeta, i)
                if layer and layer.layerName == "output0":
                    detections.extend(parse_yolo_output(layer, CONF_THRESHOLD, NMS_IOU))
        l_user = l_user.next

    payload = {"timestamp": time.time(), "detections": detections, "source": "deepstream_trt_yolo11"}
    print(f"[deepstream] publishing {len(detections)} detections to {OBS_TOPIC}")
    u_data["mqtt"].publish(OBS_TOPIC, json.dumps(payload))
    return Gst.PadProbeReturn.OK


def build_pipeline():
    Gst.init(None)
    pipeline = Gst.Pipeline()

    # Source branch: HTTP MJPEG (debug stream) or V4L2
    if VIDEO_SOURCE.startswith("http"):
        src = Gst.ElementFactory.make("souphttpsrc", "source")
        src.set_property("location", VIDEO_SOURCE)
        src.set_property("is-live", True)
        src.set_property("do-timestamp", True)
        demux = Gst.ElementFactory.make("multipartdemux", "demux")
        jpeg_caps = Gst.ElementFactory.make("capsfilter", "jpeg_caps")
        jpeg_caps.set_property("caps", Gst.Caps.from_string(f"image/jpeg,framerate={VIDEO_FPS}/1"))
        jpegdec = Gst.ElementFactory.make("jpegdec", "jpegdec")
        src_chain = [src, demux, jpeg_caps, jpegdec]
    else:
        src = Gst.ElementFactory.make("v4l2src", "source")
        src.set_property("device", V4L2_DEVICE)
        caps_src = Gst.ElementFactory.make("capsfilter", "source_caps")
        caps_src.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=YUY2,width={VIDEO_WIDTH},height={VIDEO_HEIGHT},framerate={VIDEO_FPS}/1"
            ),
        )
        conv = Gst.ElementFactory.make("videoconvert", "conv")
        src_chain = [src, caps_src, conv]

    caps_nv12 = Gst.ElementFactory.make("capsfilter", "caps_nv12")
    caps_nv12.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=NV12,framerate={VIDEO_FPS}/1"))

    conv_http = Gst.ElementFactory.make("videoconvert", "conv_http")
    nvvidconv = Gst.ElementFactory.make("nvvidconv", "nvvidconv")
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "caps_nvmm")
    caps_nvmm.set_property("caps", Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format=NV12,width=416,height=416,framerate={VIDEO_FPS}/1"))

    mux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    mux.set_property("batch-size", 1)
    mux.set_property("width", 416)
    mux.set_property("height", 416)
    mux.set_property("batched-push-timeout", 4000000)
    mux.set_property("live-source", 1)
    mux.set_property("enable-padding", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", ENGINE_CONFIG)

    queue = Gst.ElementFactory.make("queue", "queue")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    sink.set_property("sync", False)

    elems = src_chain + ([conv_http] if VIDEO_SOURCE.startswith("http") else []) + [caps_nv12, nvvidconv, caps_nvmm, mux, pgie, queue, sink]
    for elem in elems:
        if not elem:
            raise RuntimeError("Failed to create element")
        pipeline.add(elem)

    if VIDEO_SOURCE.startswith("http"):
        def on_pad_added(mdemux, pad):
            sink_pad = jpeg_caps.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)

        demux.connect("pad-added", on_pad_added)
        if not (src.link(demux) and jpeg_caps.link(jpegdec) and jpegdec.link(conv_http) and conv_http.link(caps_nv12) and caps_nv12.link(nvvidconv) and nvvidconv.link(caps_nvmm)):
            raise RuntimeError("Failed to link HTTP source elements")
    else:
        if not (src.link(src_chain[1]) and src_chain[1].link(src_chain[2]) and src_chain[2].link(caps_nv12) and caps_nv12.link(nvvidconv) and nvvidconv.link(caps_nvmm)):
            raise RuntimeError("Failed to link V4L2 source elements")
    sinkpad = mux.get_request_pad("sink_0")
    srcpad = caps_nvmm.get_static_pad("src")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link source to mux")

    if not (mux.link(pgie) and pgie.link(queue) and queue.link(sink)):
        raise RuntimeError("Failed to link mux -> pgie -> sink")

    return pipeline, pgie


def main():
    mqtt_client = mqtt.Client(client_id="deepstream_mqtt", protocol=mqtt.MQTTv311)
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, 30)
    mqtt_client.loop_start()

    if not os.path.exists(ENGINE_CONFIG):
        raise FileNotFoundError(f"ENGINE_CONFIG not found: {ENGINE_CONFIG}")
    print(f"Using nvinfer config: {ENGINE_CONFIG}")

    pipeline, pgie = build_pipeline()
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        raise RuntimeError("Unable to get src pad of nvinfer")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, infer_probe, {"mqtt": mqtt_client})

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret not in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC):
        raise RuntimeError(f"Failed to start pipeline, state change returned {ret}")
    print("DeepStream pipeline started")

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GObject.MainLoop()

    def on_message(bus, message, udata):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[deepstream] EOS received")
            udata.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[deepstream] ERROR: {err} {dbg}")
            udata.quit()

    bus.connect("message", on_message, loop)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()
