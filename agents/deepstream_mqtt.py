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
import subprocess
from typing import Dict, List, Optional, Set

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
OBS_TOPIC = os.getenv("PERCEPTION_TOPIC") or os.getenv("VISION_OBSERVATION_TOPIC", "vision/observation")

# Source selection
DS_SOURCE = os.getenv("DS_SOURCE", os.getenv("VIDEO_SOURCE", "v4l2")).lower()
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")  # used when DS_SOURCE=http
V4L2_DEVICE = os.getenv("DS_V4L2_DEVICE", os.getenv("VIDEO_DEVICE", "/dev/video0"))
V4L2_WIDTH = int(os.getenv("DS_V4L2_WIDTH", os.getenv("VIDEO_WIDTH", "0")))
V4L2_HEIGHT = int(os.getenv("DS_V4L2_HEIGHT", os.getenv("VIDEO_HEIGHT", "0")))
V4L2_FPS = int(os.getenv("DS_V4L2_FPS", os.getenv("VIDEO_FPS", "0")))
V4L2_FORMAT = os.getenv("DS_V4L2_FORMAT", "AUTO").upper()  # AUTO to probe, else MJPG/YUYV

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.6"))
CLASS_WHITELIST_RAW = os.getenv("CLASS_WHITELIST", "")
CLASS_LABELS_RAW = os.getenv("CLASS_LABELS", "")
MAX_DETECTIONS_PER_FRAME = int(os.getenv("MAX_DETECTIONS_PER_FRAME", "0"))
ENGINE_CONFIG = os.getenv("ENGINE_CONFIG", "/app/docker/deepstream/yolo11_pgie.txt")
# Match streammux/nvinfer resolution to the engine. Default 320 aligns with yolov8s_worldv2_320_fp16.engine.
ENGINE_INPUT_SIZE = int(os.getenv("ENGINE_INPUT_SIZE", "320"))
DEBUG_YOLO = os.getenv("DEBUG_YOLO", "0") == "1"
DS_V4L2_RETRY = os.getenv("DS_V4L2_RETRY", "0") == "1"
DS_V4L2_RETRY_COUNT = int(os.getenv("DS_V4L2_RETRY_COUNT", "3"))
DS_V4L2_RETRY_SLEEP = float(os.getenv("DS_V4L2_RETRY_SLEEP", "2.0"))
_seen_layers: set[str] = set()


def _parse_class_whitelist(raw: str) -> Optional[Set[int]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                out = {int(v) for v in data if str(v).strip().isdigit()}
                return out or None
        except Exception:
            pass
    tokens = [t.strip() for t in raw.replace(";", ",").split(",")]
    out = {int(t) for t in tokens if t.isdigit()}
    return out or None


def _parse_class_labels(raw: str) -> Dict[int, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{") or raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return {i: str(v) for i, v in enumerate(data) if str(v).strip()}
            if isinstance(data, dict):
                out = {}
                for k, v in data.items():
                    try:
                        idx = int(k)
                    except (TypeError, ValueError):
                        continue
                    label = str(v).strip()
                    if label:
                        out[idx] = label
                return out
        except Exception:
            return {}
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    return {i: name for i, name in enumerate(labels)}


CLASS_WHITELIST = _parse_class_whitelist(CLASS_WHITELIST_RAW)
CLASS_LABELS = _parse_class_labels(CLASS_LABELS_RAW)


def probe_v4l2_formats(device: str) -> list[str]:
    """Return lines of v4l2-ctl --list-formats-ext output or [] on failure."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"], stderr=subprocess.STDOUT, timeout=5
        )
        return out.decode("utf-8", errors="ignore").splitlines()
    except Exception as exc:  # pragma: no cover - probe best-effort
        print(f"[deepstream] v4l2-ctl probe failed: {exc}")
        return []


def select_v4l2_caps(device: str) -> tuple[str, int, int, int]:
    """
    Auto-select a safe raw format and mode. Preference:
    - YUYV/YUY2 with a reasonable size (prefer <=1280x720), fallback first seen.
    If env overrides are set, they take precedence.
    """
    # If user explicitly set non-auto values, honor them; otherwise probe
    if V4L2_FORMAT != "AUTO" or V4L2_WIDTH > 0 or V4L2_HEIGHT > 0 or V4L2_FPS > 0:
        fmt = V4L2_FORMAT if V4L2_FORMAT != "AUTO" else "YUY2"
        w = V4L2_WIDTH if V4L2_WIDTH > 0 else 1280
        h = V4L2_HEIGHT if V4L2_HEIGHT > 0 else 720
        fps = V4L2_FPS if V4L2_FPS > 0 else 30
        print(f"[deepstream] using env-specified v4l2 caps format={fmt} {w}x{h}@{fps}")
        return fmt, w, h, fps

    lines = probe_v4l2_formats(device)
    best = None
    preferred = None
    cur_fmt = None
    for line in lines:
        line = line.strip()
        if line.startswith("Pixel Format"):
            if "'" in line:
                cur_fmt = line.split("'")[1]
            else:
                cur_fmt = None
        if "Size" in line and "Discrete" in line and cur_fmt:
            parts = line.split()
            wh = parts[-1]
            if "x" in wh:
                try:
                    w, h = [int(x) for x in wh.split("x")]
                except Exception:
                    continue
                fps = 30
                if "(" in line and "fps" in line:
                    try:
                        fps = int(line.split("(")[1].split()[0].split("/")[0])
                    except Exception:
                        fps = 30
                cand = (cur_fmt, w, h, fps)
                if cur_fmt in ("YUYV", "YUY2"):
                    if not best:
                        best = cand
                    if w <= 1280 and h <= 720 and fps >= 30 and preferred is None:
                        preferred = cand
                elif not best:
                    best = cand
    pick = preferred or best
    if pick:
        fmt, w, h, fps = pick
        print(f"[deepstream] auto-selected v4l2 caps format={fmt} {w}x{h}@{fps}")
        return fmt, w, h, fps
    print("[deepstream] auto-select fallback format=YUY2 1280x720@30")
    return "YUY2", 1280, 720, 30


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


def _get_layer_ptr(layer: pyds.NvDsInferLayerInfo):
    buf = layer.buffer
    try:
        if hasattr(pyds, "get_ptr"):
            buf = pyds.get_ptr(buf)
        return ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))
    except Exception as exc:
        if DEBUG_YOLO:
            print(f"[deepstream] buffer cast failed: {exc} type={type(layer.buffer)}")
        return None


def parse_yolo_output(layer: pyds.NvDsInferLayerInfo, conf_thresh: float, iou_thresh: float) -> List[dict]:
    dims = layer.inferDims
    dims_list = [dims.d[i] for i in range(dims.numDims)]
    if len(dims_list) == 2:
        dims_list = [1, dims_list[0], dims_list[1]]
    if len(dims_list) < 3:
        return []
    if len(dims_list) > 3:
        dims_list = [dims_list[0], dims_list[1], int(np.prod(dims_list[2:]))]
    d0, d1, d2 = dims_list
    size = d0 * d1 * d2
    ptr = _get_layer_ptr(layer)
    if ptr is None:
        return []
    data = np.ctypeslib.as_array(ptr, shape=(size,))
    data = data.reshape((d0, d1, d2))
    # Normalize to shape (1, C, N) where C ~ 84.
    if d1 > d2 and d2 <= 256:
        data = data.transpose(0, 2, 1)
        d0, d1, d2 = data.shape
    elif d1 > 256 and d2 <= 256:
        data = data.transpose(0, 2, 1)
        d0, d1, d2 = data.shape

    boxes_xyxy = []
    scores = []
    classes = []
    channels = data.shape[1]
    for i in range(d2):
        xywh = data[0, 0:4, i]
        if channels >= 85:
            obj_logits = data[0, 4, i]
            cls_logits = data[0, 5:, i]
            if cls_logits.size == 0:
                continue
            if cls_logits.max() <= 1.0 and cls_logits.min() >= 0.0:
                cls_scores = cls_logits
            else:
                cls_scores = 1 / (1 + np.exp(-cls_logits))
            if obj_logits <= 1.0 and obj_logits >= 0.0:
                obj_score = float(obj_logits)
            else:
                obj_score = float(1 / (1 + math.exp(-float(obj_logits))))
            cls_id = int(cls_scores.argmax())
            score = float(cls_scores[cls_id]) * obj_score
        else:
            cls_logits = data[0, 4:, i]
            if cls_logits.size == 0:
                continue
            if cls_logits.max() <= 1.0 and cls_logits.min() >= 0.0:
                cls_scores = cls_logits
            else:
                cls_scores = 1 / (1 + np.exp(-cls_logits))
            cls_id = int(cls_scores.argmax())
            score = float(cls_scores[cls_id])
        if CLASS_WHITELIST is not None and cls_id not in CLASS_WHITELIST:
            continue
        if score < conf_thresh:
            continue
        boxes_xyxy.append(xywh_to_xyxy(xywh))
        scores.append(score)
        classes.append(cls_id)

    if not boxes_xyxy:
        if DEBUG_YOLO:
            max_score = float(cls_scores.max()) if "cls_scores" in locals() else 0.0
            print(f"[deepstream] no boxes; layer {layer.layerName} dims={dims.d[:dims.numDims]} max_score={max_score:.4f}")
        return []
    boxes_xyxy = np.stack(boxes_xyxy, axis=0)
    scores_np = np.array(scores)
    keep = nms(boxes_xyxy, scores_np, iou_thresh)
    if MAX_DETECTIONS_PER_FRAME > 0 and len(keep) > MAX_DETECTIONS_PER_FRAME:
        keep = sorted(keep, key=lambda idx: scores[idx], reverse=True)[:MAX_DETECTIONS_PER_FRAME]
    results = []
    for k in keep:
        x1, y1, x2, y2 = boxes_xyxy[k]
        class_id = int(classes[k])
        entry = {
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[k]),
            "class_id": class_id,
        }
        label = CLASS_LABELS.get(class_id)
        if label is not None:
            entry["label"] = label
        results.append(entry)
    return results


def infer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK
    print("[deepstream] frame received at pgie src")
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    def _collect_tensor_meta(user_list) -> List[dict]:
        local_detections: List[dict] = []
        l_user = user_list
        while l_user:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tmeta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                for i in range(tmeta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tmeta, i)
                    if not layer:
                        continue
                    if DEBUG_YOLO and layer.layerName not in _seen_layers:
                        print(f"[deepstream] saw layer name={layer.layerName} dims={layer.inferDims.d[:layer.inferDims.numDims]}")
                        _seen_layers.add(layer.layerName)
                    if layer.layerName == "output0":
                        local_detections.extend(parse_yolo_output(layer, CONF_THRESHOLD, NMS_IOU))
            l_user = l_user.next
        return local_detections

    detections: List[dict] = []
    detections.extend(_collect_tensor_meta(batch_meta.batch_user_meta_list))
    if not detections:
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            detections.extend(_collect_tensor_meta(frame_meta.frame_user_meta_list))
            l_frame = l_frame.next
        if DEBUG_YOLO and not detections:
            print("[deepstream] no tensor meta found on batch/frame lists")

    payload = {"timestamp": time.time(), "detections": detections, "source": "deepstream_trt_yolo11"}
    print(f"[deepstream] publishing {len(detections)} detections to {OBS_TOPIC}")
    u_data["mqtt"].publish(OBS_TOPIC, json.dumps(payload))
    return Gst.PadProbeReturn.OK


def build_pipeline():
    Gst.init(None)
    pipeline = Gst.Pipeline()

    source_mode = DS_SOURCE or "v4l2"
    fmt, width, height, fps = (
        select_v4l2_caps(V4L2_DEVICE) if source_mode == "v4l2" else (V4L2_FORMAT, V4L2_WIDTH, V4L2_HEIGHT, V4L2_FPS)
    )
    print(f"[deepstream] source mode={source_mode}, device={V4L2_DEVICE}, format={fmt}, size={width}x{height}@{fps}")

    elems = []

    if source_mode == "http" or (source_mode == "auto" and VIDEO_SOURCE.startswith("http")):
        src = Gst.ElementFactory.make("souphttpsrc", "source")
        src.set_property("location", VIDEO_SOURCE)
        src.set_property("is-live", True)
        src.set_property("do-timestamp", True)
        demux = Gst.ElementFactory.make("multipartdemux", "demux")
        jpeg_caps = Gst.ElementFactory.make("capsfilter", "jpeg_caps")
        jpeg_caps.set_property("caps", Gst.Caps.from_string(f"image/jpeg,framerate={V4L2_FPS}/1"))
        jpegdec = Gst.ElementFactory.make("jpegdec", "jpegdec")
        conv_http = Gst.ElementFactory.make("videoconvert", "conv_http")
        elems.extend([src, demux, jpeg_caps, jpegdec, conv_http])
        http_chain = True
    else:
        # V4L2 source
        src = Gst.ElementFactory.make("v4l2src", "source")
        src.set_property("device", V4L2_DEVICE)
        http_chain = False
        if fmt == "MJPG":
            caps_src = Gst.ElementFactory.make("capsfilter", "source_caps")
            caps_src.set_property("caps", Gst.Caps.from_string(f"image/jpeg,framerate={fps}/1"))
            jpegparse = Gst.ElementFactory.make("jpegparse", "jpegparse")
            decoder = Gst.ElementFactory.make("nvv4l2decoder", "mjpegdecoder") or Gst.ElementFactory.make("jpegdec", "jpegdec")
            elems.extend([src, caps_src, jpegparse, decoder])
        else:
            conv = Gst.ElementFactory.make("videoconvert", "conv")
            scale = Gst.ElementFactory.make("videoscale", "videoscale")
            target_caps = Gst.ElementFactory.make("capsfilter", "target_caps")
            target_caps.set_property("caps", Gst.Caps.from_string(f"video/x-raw,width={width},height={height},framerate={fps}/1"))
            elems.extend([src, conv, scale, target_caps])

    # Common downstream: convert to NV12 NVMM and feed streammux
    nvvidconv = Gst.ElementFactory.make("nvvidconv", "nvvidconv")
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "caps_nvmm")
    caps_nvmm.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=NV12,width={ENGINE_INPUT_SIZE},height={ENGINE_INPUT_SIZE},framerate={fps}/1"
        ),
    )

    mux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    mux.set_property("batch-size", 1)
    mux.set_property("width", ENGINE_INPUT_SIZE)
    mux.set_property("height", ENGINE_INPUT_SIZE)
    mux.set_property("batched-push-timeout", 4000000)
    mux.set_property("live-source", 1)
    mux.set_property("enable-padding", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", ENGINE_CONFIG)

    queue = Gst.ElementFactory.make("queue", "queue")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    sink.set_property("sync", False)

    elems += [nvvidconv, caps_nvmm, mux, pgie, queue, sink]
    for elem in elems:
        if not elem:
            raise RuntimeError("Failed to create element")
        pipeline.add(elem)

    if http_chain:
        def on_pad_added(mdemux, pad):
            sink_pad = jpeg_caps.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)

        demux.connect("pad-added", on_pad_added)
        if not (src.link(demux) and jpeg_caps.link(jpegdec) and jpegdec.link(conv_http) and conv_http.link(nvvidconv) and nvvidconv.link(caps_nvmm)):
            raise RuntimeError("Failed to link HTTP source elements")
    else:
        if fmt == "MJPG":
            if not (src.link(caps_src) and caps_src.link(jpegparse) and jpegparse.link(decoder) and decoder.link(nvvidconv) and nvvidconv.link(caps_nvmm)):
                raise RuntimeError("Failed to link V4L2 MJPG elements")
        else:
            if not (src.link(conv) and conv.link(scale) and scale.link(target_caps) and target_caps.link(nvvidconv) and nvvidconv.link(caps_nvmm)):
                raise RuntimeError("Failed to link V4L2 YUYV elements")

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

    attempt = 0
    while True:
        attempt += 1
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
        error_state: Optional[dict] = None

        def on_message(bus, message, udata):
            nonlocal error_state
            t = message.type
            if t == Gst.MessageType.EOS:
                print("[deepstream] EOS received")
                udata.quit()
            elif t == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                print(f"[deepstream] ERROR: {err} {dbg}")
                if "Device or resource busy" in str(err):
                    print("[deepstream] V4L2 device appears BUSY or stuck. Run tools/camera_debug.sh, then tools/camera_reset.sh, then retry.")
                error_state = {"err": err, "dbg": dbg}
                udata.quit()

        bus.connect("message", on_message, loop)
        try:
            loop.run()
        except KeyboardInterrupt:
            pipeline.set_state(Gst.State.NULL)
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            return
        finally:
            pipeline.set_state(Gst.State.NULL)

        if error_state and DS_V4L2_RETRY and attempt < DS_V4L2_RETRY_COUNT:
            print(f"[deepstream] retrying in {DS_V4L2_RETRY_SLEEP}s (attempt {attempt}/{DS_V4L2_RETRY_COUNT})...")
            time.sleep(DS_V4L2_RETRY_SLEEP)
            continue

        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        if error_state:
            raise RuntimeError(f"Pipeline stopped due to error: {error_state['err']}")
        break


if __name__ == "__main__":
    main()
