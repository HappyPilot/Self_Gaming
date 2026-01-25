#!/usr/bin/env python3
import io
import hashlib
import json
import logging
import os
import socket
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_JIT", "0")

import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from utils.frame_transport import get_frame_bytes
from utils.latency import emit_latency, get_sla_ms

try:
    import cv2
except ImportError:
    cv2 = None
if cv2 is not None:
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None
if torch is not None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# --- Setup ---
logging.basicConfig(level=os.getenv("OCR_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ocr_easy_agent")
stop_event = threading.Event()

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
OCR_CLIENT_ID = os.getenv("OCR_CLIENT_ID")
VISION_CMD = os.getenv("VISION_CMD", "vision/cmd")
VISION_SNAPSHOT = os.getenv("VISION_SNAPSHOT", "vision/snapshot")
OCR_FRAME_TOPIC = os.getenv("OCR_FRAME_TOPIC", "").strip()
VISION_FRAME = os.getenv("VISION_FRAME", OCR_FRAME_TOPIC or os.getenv("VISION_FRAME_TOPIC", "vision/frame/preview"))
OCR_CMD = os.getenv("OCR_CMD", "ocr_easy/cmd")
OCR_TEXT = os.getenv("OCR_TEXT", "ocr_easy/text")
OCR_LANGS = [lang.strip() for lang in os.getenv("OCR_LANGS", "en,ru").split(",") if lang.strip()]
OCR_BACKEND = os.getenv("OCR_BACKEND", "paddle").strip().lower()
SLA_STAGE_OCR_MS = get_sla_ms("SLA_STAGE_OCR_MS")
AUTO_INTERVAL = float(os.getenv("OCR_AUTO_INTERVAL", "3.0"))
AUTO_TIMEOUT = float(os.getenv("OCR_AUTO_TIMEOUT", "2.0"))
DEBUG_SAVE = os.getenv("OCR_DEBUG_SAVE", "0") == "1"
DEBUG_DIR = Path(os.getenv("OCR_DEBUG_DIR", "/tmp/ocr_debug"))
FORCE_CPU = os.getenv("OCR_FORCE_CPU", "0") == "1"
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "1").lower() not in {"0", "false", "no"}
SCALE_FACTOR = float(os.getenv("OCR_SCALE_FACTOR", "1.0"))  # keep <=1.0 to downscale
TARGET_WIDTH = float(os.getenv("OCR_MAX_BASE_WIDTH", "640"))  # max side for radar mode
GAMMA = float(os.getenv("OCR_GAMMA", "1.3"))
MIN_ALPHA_RATIO = float(os.getenv("OCR_MIN_ALPHA_RATIO", "0.35"))
MAX_LINE_LENGTH = int(os.getenv("OCR_MAX_LINE_CHARS", "160"))
VARIANT_THRESHOLD = os.getenv("OCR_VARIANT_THRESHOLD", "adaptive")
MAX_RESULTS = int(os.getenv("OCR_MAX_RESULTS", "12"))
PREFER_SNAPSHOT = os.getenv("OCR_PREFER_SNAPSHOT", "1") != "0"
FALLBACK_ON_EMPTY = os.getenv("OCR_FALLBACK_ON_EMPTY", "0") == "1"
FALLBACK_MIN_RESULTS = max(1, int(os.getenv("OCR_FALLBACK_MIN_RESULTS", "1")))
FALLBACK_BACKEND = os.getenv("OCR_FALLBACK_BACKEND", "easyocr").strip().lower()

# Frame hash gating (skip OCR on unchanged frames)
HASH_ENABLE = os.getenv("OCR_HASH_ENABLE", "1") == "1"
HASH_SIZE = int(os.getenv("OCR_HASH_SIZE", "160"))
HASH_FORCE_SEC = float(os.getenv("OCR_HASH_FORCE_SEC", "30.0"))

last_frame_hash: Optional[str] = None
last_ocr_ts: float = 0.0
MAX_RESULTS = int(os.getenv("OCR_MAX_RESULTS", "12"))

if FORCE_CPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0


def _frame_hash(img: Image.Image) -> str:
    """Compute a small grayscale hash for change detection."""
    if HASH_SIZE <= 0:
        return ""
    w = HASH_SIZE
    h = max(1, int(img.height * (HASH_SIZE / max(1, img.width))))
    gray = img.convert("L").resize((w, h), resample=Image.BILINEAR)
    arr = np.array(gray)
    return hashlib.md5(arr.tobytes()).hexdigest()

class OcrEasyAgent:
    def __init__(self):
        client_id = OCR_CLIENT_ID or f"ocr_easy_{socket.gethostname()}_{os.getpid()}"
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.message_callback_add(VISION_SNAPSHOT, self._on_snapshot)
        self.client.message_callback_add(VISION_FRAME, self._on_frame)
        self.client.message_callback_add(OCR_CMD, self._on_cmd)
        
        self.snap_lock = threading.Lock()
        self.snap_data = None
        self.frame_lock = threading.Lock()
        self.frame_data = None
        
        self.gpu = False
        self.reader = None
        self.rapidocr_reader = None
        self.backend = None
        self.init_error = None
        self.ready = False
        self._backend_lock = threading.Lock()
        self._backend_initialized = False
        self._ocr_lock = threading.Lock()

    def _init_backend(self):
        try:
            import torch
            self.gpu = torch.cuda.is_available() and not FORCE_CPU and OCR_USE_GPU
        except Exception:
            self.gpu = False

        if OCR_BACKEND in {"paddle", "auto"}:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.rapidocr_reader = RapidOCR(
                    det_use_cuda=not FORCE_CPU and self.gpu,
                    cls_use_cuda=not FORCE_CPU and self.gpu,
                    rec_use_cuda=not FORCE_CPU and self.gpu,
                    rec_score_thresh=0.1,
                )
                self.backend = "paddle"
            except Exception as exc:
                self.init_error = f"paddle_init_failed: {exc}"
                logger.warning("Paddle init failed: %s", exc)

        if self.backend is None and OCR_BACKEND in {"easyocr", "auto", "paddle"}:
            try:
                import easyocr
                self.reader = easyocr.Reader(OCR_LANGS or ["en"], gpu=self.gpu)
                self.backend = "easyocr"
            except Exception as exc:
                self.init_error = self.init_error or f"easyocr_init_failed: {exc}"
                logger.warning("EasyOCR init failed: %s", exc)

        self.ready = (self.backend == "paddle" and self.rapidocr_reader is not None) or \
                     (self.backend == "easyocr" and self.reader is not None)
        logger.info("OCR backend ready: %s (backend=%s, gpu=%s)", self.ready, self.backend, self.gpu)

    def _ensure_easyocr(self) -> bool:
        if self.reader is not None:
            return True
        if torch is not None:
            try:
                self.gpu = torch.cuda.is_available() and not FORCE_CPU and OCR_USE_GPU
            except Exception:
                self.gpu = False
        try:
            import easyocr
            self.reader = easyocr.Reader(OCR_LANGS or ["en"], gpu=self.gpu)
            return True
        except Exception as exc:
            logger.warning("EasyOCR fallback init failed: %s", exc)
            return False

    def _ensure_backend(self):
        if self._backend_initialized:
            return
        with self._backend_lock:
            if self._backend_initialized:
                return
            try:
                self._init_backend()
            finally:
                self._backend_initialized = True

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(VISION_SNAPSHOT, 0), (VISION_FRAME, 0), (OCR_CMD, 0)])
            client.publish(
                OCR_TEXT,
                json.dumps(
                    {"ok": True, "event": "connected", "gpu": self.gpu, "listen": [VISION_SNAPSHOT, VISION_FRAME, OCR_CMD]}
                ),
            )
            logger.info("Connected to MQTT")
        else:
            logger.error("Connect failed rc=%s", _as_int(rc))
            client.publish(OCR_TEXT, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _on_snapshot(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload)
            raw = get_frame_bytes(d)
            if not raw:
                return
            with self.snap_lock:
                self.snap_data = raw
            logger.debug("Snapshot received, bytes: %s", len(self.snap_data))
        except Exception as e:
            logger.error("Snapshot error: %s", e)

    def _on_frame(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            raw = get_frame_bytes(data)
            if not raw:
                return
            with self.frame_lock:
                self.frame_data = raw
        except Exception:
            pass

    def _on_cmd(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except Exception:
            payload = {}
        threading.Thread(target=self._handle_cmd, args=(payload,), daemon=True).start()

    def _handle_cmd(self, payload):
        self._ensure_backend()
        if not self.ready:
            self.client.publish(OCR_TEXT, json.dumps({"ok": False, "error": "easyocr_init_failed", "detail": self.init_error}))
            return
        cmd = (payload.get("cmd") or "").lower()
        if cmd == "once":
            self._process_once(payload)

    def _request_snapshot(self, timeout=2.0):
        with self.snap_lock:
            self.snap_data = None
        self.client.publish(VISION_CMD, json.dumps({"cmd": "snapshot"}))
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.snap_lock:
                if self.snap_data is not None:
                    return self.snap_data
            time.sleep(0.02)
        return None

    def _scale_image(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        scale = SCALE_FACTOR
        if TARGET_WIDTH > 0:
            scale = min(scale, TARGET_WIDTH / max(1.0, w))
        if scale <= 0 or abs(scale - 1.0) < 1e-3:
            return gray
        target = (max(1, int(w * scale)), max(1, int(h * scale)))
        if cv2 is None:
            return np.array(Image.fromarray(gray).resize(target, resample=Image.BICUBIC))
        return cv2.resize(gray, target, interpolation=cv2.INTER_CUBIC)

    def _preprocess_variants(self, img: Image.Image) -> List[np.ndarray]:
        base = ImageOps.autocontrast(img.convert("L"), cutoff=1)
        gray = np.array(base)
        if GAMMA != 1.0:
            inv_gamma = 1.0 / max(0.01, GAMMA)
            table = ((np.linspace(0, 1, 256) ** inv_gamma) * 255.0).astype("uint8")
            if cv2 is not None:
                gray = cv2.LUT(gray, table)
            else:
                lut = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
                gray = np.array(Image.fromarray(gray).point(lut))
        scaled = self._scale_image(gray)

        if cv2 is None:
            blurred = np.array(Image.fromarray(scaled).filter(ImageFilter.GaussianBlur(radius=1.5)))
            return [blurred, np.array(ImageOps.invert(Image.fromarray(blurred)))]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
        blurred = cv2.bilateralFilter(enhanced, d=7, sigmaColor=30, sigmaSpace=30)
        variants = [blurred]
        if VARIANT_THRESHOLD.lower() != "off":
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
            variants.append(thresh)
            variants.append(cv2.bitwise_not(thresh))
        variants.append(cv2.bitwise_not(blurred))
        return variants

    def _convert_to_color(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[2] == 3: return arr
        if cv2 is not None: return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return np.dstack([arr] * 3)

    def _recognize_variant(self, arr: np.ndarray, backend: Optional[str] = None) -> List[dict]:
        if not self.ready: return []
        h, w = arr.shape[:2]
        results = []

        use_backend = backend or self.backend
        if use_backend == "paddle":
            if self.rapidocr_reader is None: return []
            with self._ocr_lock:
                raw_res, _ = self.rapidocr_reader(self._convert_to_color(arr))
            if raw_res:
                for entry in raw_res:
                    # RapidOCR: [ [[x1,y1], [x2,y2], ...], text, conf ]
                    if not entry or not entry[1]: continue
                    poly, text, conf = entry
                    # Convert polygon to bbox [x, y, w, h] normalized
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    results.append({
                        "text": text,
                        "box": [x_min/w, y_min/h, (x_max-x_min)/w, (y_max-y_min)/h],
                        "conf": float(conf)
                    })

        elif use_backend == "easyocr":
            if self.reader is None: return []
            # EasyOCR: [ ([[x1,y1], ...], text, conf), ... ]
            with self._ocr_lock:
                if torch is not None:
                    with torch.no_grad():
                        raw_res = self.reader.readtext(arr)
                else:
                    raw_res = self.reader.readtext(arr)
            for poly, text, conf in raw_res:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                results.append({
                    "text": text,
                    "box": [x_min/w, y_min/h, (x_max-x_min)/w, (y_max-y_min)/h],
                    "conf": float(conf)
                })
        return results

    def _is_noise(self, fragment: str) -> bool:
        fragment = fragment.strip()
        if not fragment or len(fragment) > MAX_LINE_LENGTH: return True
        letters = sum(ch.isalpha() for ch in fragment)
        if (letters / max(1, len(fragment))) < MIN_ALPHA_RATIO and not any(ch.isdigit() for ch in fragment): return True
        return False

    def _process_once(self, payload):
        self._ensure_backend()
        if not self.ready:
            self.client.publish(OCR_TEXT, json.dumps({"ok": False, "error": self.init_error or "ocr_not_ready"}))
            return
        timeout = float(payload.get("timeout", AUTO_TIMEOUT))
        source, jpeg = "frame", None
        if PREFER_SNAPSHOT:
            source, jpeg = "snapshot", self._request_snapshot(timeout)
        if jpeg is None:
            source = "frame"
            with self.frame_lock:
                jpeg = self.frame_data
            if jpeg is None:
                self.client.publish(OCR_TEXT, json.dumps({"ok": False, "error": "no_frame", "timeout": timeout}))
                return

        ocr_start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(jpeg))
            global last_frame_hash, last_ocr_ts
            if HASH_ENABLE:
                h = _frame_hash(img)
                now = time.time()
                # Force a full OCR at least every HASH_FORCE_SEC even if unchanged
                if last_frame_hash == h and (now - last_ocr_ts) < HASH_FORCE_SEC:
                    self.client.publish(OCR_TEXT, json.dumps({"ok": True, "text": "", "results": [], "backend": self.backend, "empty": True, "skipped": "unchanged_frame"}))
                    return
                last_frame_hash = h
                last_ocr_ts = now
            
            all_results = []
            seen_texts = set()
            
            variants = self._preprocess_variants(img)
            for variant in variants:
                for res in self._recognize_variant(variant):
                    cleaned = res["text"].strip()
                    if not cleaned or self._is_noise(cleaned): continue
                    # Deduplicate roughly
                    key = f"{cleaned}_{res['box'][0]:.2f}_{res['box'][1]:.2f}"
                    if key in seen_texts: continue
                    seen_texts.add(key)
                    
                    all_results.append(res)

            if (
                FALLBACK_ON_EMPTY
                and self.backend != "easyocr"
                and FALLBACK_BACKEND == "easyocr"
                and len(all_results) < FALLBACK_MIN_RESULTS
                and self._ensure_easyocr()
                and variants
            ):
                for res in self._recognize_variant(variants[0], backend="easyocr"):
                    cleaned = res["text"].strip()
                    if not cleaned or self._is_noise(cleaned):
                        continue
                    key = f"{cleaned}_{res['box'][0]:.2f}_{res['box'][1]:.2f}"
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    all_results.append(res)

            # Keep top-N by confidence to limit payload size
            all_results.sort(key=lambda r: r.get("conf", 0.0), reverse=True)
            if MAX_RESULTS > 0:
                all_results = all_results[:MAX_RESULTS]
            
            # Construct text block for compatibility
            full_text = "\n".join([r["text"] for r in all_results])
            
            payload = {
                "ok": True, 
                "text": full_text, 
                "results": all_results, # New field with coordinates
                "backend": self.backend
            }
            self.client.publish(OCR_TEXT, json.dumps(payload))
            logger.debug("OCR published %s items", len(all_results))
        except Exception as e:
            self.client.publish(OCR_TEXT, json.dumps({"ok": False, "error": str(e)}))
            logger.error("OCR error: %s", e)
        finally:
            ocr_ms = (time.perf_counter() - ocr_start) * 1000.0
            emit_latency(
                self.client,
                "ocr",
                ocr_ms,
                sla_ms=SLA_STAGE_OCR_MS,
                tags={"backend": self.backend, "source": source},
                agent="ocr_easy_agent",
            )

    def _auto_loop(self):
        if AUTO_INTERVAL <= 0:
            return
        while not stop_event.is_set():
            self._ensure_backend()
            if self.ready:
                try:
                    self._process_once({"timeout": AUTO_TIMEOUT})
                except Exception as e:
                    self.client.publish(OCR_TEXT, json.dumps({"ok": False, "error": f"auto_loop_failed:{e}"}))
            stop_event.wait(AUTO_INTERVAL)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        threading.Thread(target=self._auto_loop, daemon=True).start()
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("OCR agent shut down")

def _handle_signal(signum, frame):
    logger.info("Signal %s received", signum)
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    OcrEasyAgent().run()

if __name__ == "__main__":
    main()
