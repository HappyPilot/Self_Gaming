#!/usr/bin/env python3
"""MQTT agent that publishes visual embeddings from incoming frames."""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import signal
import threading
import time
from typing import Optional, Sequence

import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image

from utils.frame_transport import get_frame_bytes
from utils.latency import emit_latency, get_sla_ms
from utils.trt_runner import TensorRTRunner

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None
try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # noqa: BLE001
    AutoImageProcessor = None
    AutoModel = None

logging.basicConfig(level=os.getenv("VL_JEPA_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("vl_jepa_agent")
stop_event = threading.Event()

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
FRAME_TOPIC = os.getenv("VL_JEPA_FRAME_TOPIC") or os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview")
EMBED_TOPIC = os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings")
BACKEND = os.getenv("VL_JEPA_BACKEND", "dummy").strip().lower()
MODEL_PATH = os.getenv("VL_JEPA_MODEL_PATH", "")
ENGINE_PATH = os.getenv("VL_JEPA_ENGINE_PATH", "")
DEVICE_RAW = os.getenv("VL_JEPA_DEVICE", "auto")
INPUT_SIZE = int(os.getenv("VL_JEPA_INPUT_SIZE", "224"))
EMBED_DIM = int(os.getenv("VL_JEPA_EMBED_DIM", "512"))
FRAME_STRIDE = int(os.getenv("VL_JEPA_FRAME_STRIDE", "1"))
FRAME_QUEUE_MAX = int(os.getenv("VL_JEPA_QUEUE", "2"))
FP16 = os.getenv("VL_JEPA_FP16", "0") not in {"0", "false", "False"}
MEAN_STR = os.getenv("VL_JEPA_MEAN", "")
STD_STR = os.getenv("VL_JEPA_STD", "")
SIGLIP_MODEL_ID = os.getenv("VL_JEPA_MODEL_ID", "").strip()
SIGLIP_POOL = os.getenv("VL_JEPA_POOL", "pooler").strip().lower()
SIGLIP_ATTN_IMPL = os.getenv("VL_JEPA_ATTN_IMPL", "").strip().lower()
MAX_EMBED_DIM = int(os.getenv("VL_JEPA_MAX_EMBED_DIM", "4096"))
MAX_PAYLOAD_BYTES = int(os.getenv("VL_JEPA_MAX_PAYLOAD_BYTES", "262144"))
SLA_STAGE_EMBED_MS = get_sla_ms("SLA_STAGE_EMBED_MS")


def _parse_float_list(raw: str, fallback: Sequence[float], expected_len: int) -> list[float]:
    if not raw:
        return list(fallback)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if expected_len and len(parts) != expected_len:
        return list(fallback)
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return list(fallback)
    return values


def _resolve_device(device: str) -> str:
    if device and device.lower() not in {"auto", "default"}:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BaseEncoder:
    backend = "base"

    def encode(self, image: Image.Image) -> Optional[np.ndarray]:
        raise NotImplementedError


class DummyEncoder(BaseEncoder):
    backend = "dummy"

    def __init__(self, dim: int):
        self.dim = max(1, dim)

    def encode(self, _image: Image.Image) -> Optional[np.ndarray]:
        return np.zeros(self.dim, dtype=np.float32)


class TorchScriptEncoder(BaseEncoder):
    backend = "torchscript"

    def __init__(self, model_path: str, device: str, input_size: int, mean: Sequence[float], std: Sequence[float], fp16: bool):
        if torch is None:
            raise RuntimeError("torch not available")
        if not model_path:
            raise ValueError("VL_JEPA_MODEL_PATH must be set for torchscript backend")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VL_JEPA_MODEL_PATH not found: {model_path}")
        self.device = device
        self.input_size = max(1, input_size)
        self.fp16 = fp16 and device.startswith("cuda")
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

    def encode(self, image: Image.Image) -> Optional[np.ndarray]:
        if torch is None:
            return None
        tensor = self._prepare_tensor(image)
        if self.fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            output = self.model(tensor)
        embedding = _extract_embedding(output)
        if embedding is None:
            return None
        return embedding.astype(np.float32)

    def _prepare_tensor(self, image: Image.Image) -> "torch.Tensor":
        resized = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor.to(self.device)


class TensorRTEncoder(BaseEncoder):
    backend = "tensorrt"

    def __init__(self, engine_path: str, input_size: int, mean: Sequence[float], std: Sequence[float]):
        if not engine_path:
            raise ValueError("VL_JEPA_ENGINE_PATH must be set for tensorrt backend")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"VL_JEPA_ENGINE_PATH not found: {engine_path}")
        self.input_size = max(1, input_size)
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.runner = TensorRTRunner(
            engine_path,
            input_shape=(1, 3, self.input_size, self.input_size),
            logger=logger,
        )
        self.input_dtype = self.runner.input_dtype

    def _prepare_tensor(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))[None, ...].copy(order="C")
        if self.input_dtype != np.float32:
            arr = arr.astype(self.input_dtype)
        return arr

    def encode(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            inp = self._prepare_tensor(image)
            return self.runner.infer(inp)
        except Exception:  # noqa: BLE001
            logger.exception("TensorRT inference failed")
            return None


class SiglipEncoder(BaseEncoder):
    backend = "siglip2"

    def __init__(self, model_id: str, device: str, input_size: int, fp16: bool, pool: str, attn_impl: str):
        if torch is None or AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError("transformers/torch not available for siglip2 backend")
        self.device = device
        self.input_size = max(1, input_size)
        self.fp16 = fp16 and device.startswith("cuda")
        self.pool = pool or "pooler"
        self.attn_impl = attn_impl or ""
        self.model_id = model_id or "google/siglip2-base-patch16-224"

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self):
        kwargs = {}
        if self.fp16:
            kwargs["torch_dtype"] = torch.float16
        if self.attn_impl:
            kwargs["attn_implementation"] = self.attn_impl
        try:
            return AutoModel.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            return AutoModel.from_pretrained(self.model_id, **kwargs)

    def _prepare_inputs(self, image: Image.Image):
        if self.input_size:
            image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return None
        pixel_values = pixel_values.to(self.device)
        if self.fp16:
            pixel_values = pixel_values.half()
        return {"pixel_values": pixel_values}

    def _pool_output(self, outputs):
        if self.pool in {"image_embeds", "image"} and hasattr(outputs, "image_embeds"):
            return outputs.image_embeds
        if self.pool in {"pooler", "pool"} and hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            last = outputs.last_hidden_state
            if self.pool in {"mean", "avg"}:
                return last.mean(dim=1)
            return last[:, 0]
        if isinstance(outputs, (list, tuple)) and outputs:
            return outputs[0]
        if isinstance(outputs, dict):
            for key in ("pooler_output", "image_embeds", "last_hidden_state"):
                if key in outputs:
                    return outputs[key]
        return None

    def encode(self, image: Image.Image) -> Optional[np.ndarray]:
        if torch is None:
            return None
        inputs = self._prepare_inputs(image)
        if not inputs:
            return None
        with torch.inference_mode():
            outputs = self.model(**inputs)
        pooled = self._pool_output(outputs)
        if pooled is None:
            return None
        if pooled.ndim >= 2:
            pooled = pooled[0]
        return pooled.float().cpu().numpy()


def _extract_embedding(output) -> Optional[np.ndarray]:
    if torch is None:
        return None
    tensor = None
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, dict):
        for key in ("embedding", "embeddings", "features", "feat", "output"):
            candidate = output.get(key)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
    elif isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                tensor = item
                break
    if tensor is None:
        return None
    tensor = tensor.detach()
    if tensor.ndim >= 2:
        tensor = tensor.reshape(tensor.shape[0], -1)[0]
    return tensor.float().cpu().numpy()


class VlJepaAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="vl_jepa_agent", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.encoder = None
        self.encoder_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.frame_stride = max(1, FRAME_STRIDE)
        self._frame_counter = 0

    def start(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        self.worker_thread.start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self.worker_thread.join(timeout=2)
        logger.info("VL-JEPA agent shut down.")

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(FRAME_TOPIC, 0)])
            logger.info("Subscribed to %s", FRAME_TOPIC)
            client.publish(EMBED_TOPIC, json.dumps({"ok": True, "event": "vl_jepa_ready"}))
        else:
            logger.error("MQTT connect failed: rc=%s", rc)
            client.publish(EMBED_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": rc}))

    def _on_disconnect(self, _client, _userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected: rc=%s", rc)

    def _on_message(self, _client, _userdata, msg):
        self._frame_counter += 1
        if self.frame_stride > 1 and self._frame_counter % self.frame_stride != 0:
            return
        try:
            self.frame_queue.put_nowait(msg.payload)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(msg.payload)
            except queue.Full:
                logger.warning("Frame queue full, dropping frame.")

    def _ensure_encoder(self):
        if self.encoder is not None:
            return
        with self.encoder_lock:
            if self.encoder is not None:
                return
            device = _resolve_device(DEVICE_RAW)
            mean = _parse_float_list(MEAN_STR, [0.485, 0.456, 0.406], 3)
            std = _parse_float_list(STD_STR, [0.229, 0.224, 0.225], 3)
            encoder = None
            if BACKEND == "torchscript":
                if torch is None:
                    logger.warning("VL_JEPA_BACKEND=torchscript but torch is unavailable; falling back to dummy.")
                else:
                    try:
                        encoder = TorchScriptEncoder(MODEL_PATH, device, INPUT_SIZE, mean, std, FP16)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("TorchScript init failed: %s", exc)
            elif BACKEND in {"tensorrt", "trt"}:
                try:
                    encoder = TensorRTEncoder(ENGINE_PATH, INPUT_SIZE, mean, std)
                except Exception as exc:  # noqa: BLE001
                    logger.error("TensorRT init failed: %s", exc)
            elif BACKEND in {"siglip", "siglip2"}:
                try:
                    encoder = SiglipEncoder(
                        SIGLIP_MODEL_ID,
                        device,
                        INPUT_SIZE,
                        FP16,
                        SIGLIP_POOL,
                        SIGLIP_ATTN_IMPL,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("SigLIP init failed: %s", exc)
            elif BACKEND not in {"dummy", ""}:
                logger.warning("Unknown VL_JEPA_BACKEND=%s; falling back to dummy", BACKEND)
            if encoder is None:
                encoder = DummyEncoder(EMBED_DIM)
            self.encoder = encoder
            if isinstance(self.encoder, SiglipEncoder):
                logger.info(
                    "VL-JEPA encoder ready: backend=%s device=%s input=%s model=%s",
                    self.encoder.backend,
                    device,
                    INPUT_SIZE,
                    self.encoder.model_id,
                )
            else:
                logger.info("VL-JEPA encoder ready: backend=%s device=%s input=%s", self.encoder.backend, device, INPUT_SIZE)

    def _worker_loop(self):
        while not stop_event.is_set():
            try:
                payload_bytes = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                payload = json.loads(payload_bytes.decode("utf-8", "ignore"))
            except Exception:
                continue
            frame_bytes = get_frame_bytes(payload)
            if not frame_bytes:
                continue
            image = _decode_image(frame_bytes)
            if image is None:
                continue
            self._ensure_encoder()
            frame_ts = payload.get("frame_ts") or payload.get("timestamp")
            embed_start = time.perf_counter()
            embedding = self.encoder.encode(image) if self.encoder else None
            embed_ms = (time.perf_counter() - embed_start) * 1000.0
            emit_latency(
                self.client,
                "embed",
                embed_ms,
                sla_ms=SLA_STAGE_EMBED_MS,
                tags={"frame_ts": frame_ts},
                agent="vl_jepa_agent",
            )
            if embedding is None:
                continue
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            if MAX_EMBED_DIM and len(embedding_list) > MAX_EMBED_DIM:
                logger.warning("Embedding dim too large (%s > %s); dropping payload.", len(embedding_list), MAX_EMBED_DIM)
                continue
            if EMBED_DIM and len(embedding_list) != EMBED_DIM:
                logger.warning("Embedding dim mismatch: got=%s expected=%s", len(embedding_list), EMBED_DIM)
            out_payload = {
                "ok": True,
                "timestamp": time.time(),
                "frame_ts": frame_ts or time.time(),
                "frame_id": payload.get("frame_id"),
                "embedding": embedding_list,
                "dim": len(embedding_list),
                "backend": self.encoder.backend if self.encoder else "unknown",
                "model_path": MODEL_PATH,
                "model_id": getattr(self.encoder, "model_id", ""),
            }
            payload_json = json.dumps(out_payload)
            if MAX_PAYLOAD_BYTES and len(payload_json) > MAX_PAYLOAD_BYTES:
                logger.warning("Embedding payload too large (%s bytes > %s); dropping.", len(payload_json), MAX_PAYLOAD_BYTES)
                continue
            self.client.publish(EMBED_TOPIC, payload_json)


def _decode_image(data: bytes) -> Optional[Image.Image]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB")
    except Exception:
        return None


def _handle_signal(signum, frame):
    logger.info("Signal received, shutting down.")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = VlJepaAgent()
    agent.start()


if __name__ == "__main__":
    main()
