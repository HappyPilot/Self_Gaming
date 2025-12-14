import os, io, json, base64, threading, time
from pathlib import Path
from PIL import Image
import numpy as np
import paho.mqtt.client as mqtt

# ---- конфиг из ENV ----
MQTT_HOST = os.getenv("MQTT_HOST","127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT","1883"))
VISION_CMD = os.getenv("VISION_CMD","vision/cmd")
VISION_SNAPSHOT = os.getenv("VISION_SNAPSHOT","vision/snapshot")
VISION_FRAME = os.getenv("VISION_FRAME","vision/frame")
OCR_CMD = os.getenv("OCR_CMD","ocr_easy/cmd")
OCR_TEXT = os.getenv("OCR_TEXT","ocr_easy/text")
OCR_LANGS = os.getenv("OCR_LANGS","en,ru").split(",")
AUTO_INTERVAL = float(os.getenv("OCR_AUTO_INTERVAL", "2.5"))
AUTO_TIMEOUT = float(os.getenv("OCR_AUTO_TIMEOUT", "2.0"))
DEBUG_SAVE = os.getenv("OCR_DEBUG_SAVE", "0") == "1"
DEBUG_DIR = Path(os.getenv("OCR_DEBUG_DIR", "/tmp/ocr_debug"))
FORCE_CPU = os.getenv("OCR_FORCE_CPU", "0") == "1"
if FORCE_CPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# ---- EasyOCR init (GPU auto) ----
gpu = False
try:
    import torch
    gpu = torch.cuda.is_available() and not FORCE_CPU
except Exception:
    gpu = False

try:
    import easyocr
    reader = easyocr.Reader([l.strip() for l in OCR_LANGS if l.strip()], gpu=gpu)
    ready = True
except Exception as e:
    reader = None
    ready = False
    init_error = str(e)

# ---- snapshot latch ----
snap_lock = threading.Lock()
snap_data = None
frame_lock = threading.Lock()
frame_data = None


def debug_log(event: str, **extra):
    if not DEBUG_SAVE:
        return
    entry = {"ok": True, "event": f"ocr_easy_{event}"}
    entry.update(extra)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

def on_snapshot(client, userdata, msg):
    global snap_data
    try:
        d = json.loads(msg.payload)
        b64 = d.get("image_b64")
        if not b64:
            return
        with snap_lock:
            snap_data = base64.b64decode(b64)
        print(json.dumps({"ok": True, "event": "snapshot_received", "bytes": len(snap_data)}), flush=True)
    except Exception:
        pass

def request_snapshot(cli, timeout=2.0):
    """Запросить кадр у vision и вернуть bytes JPEG или None."""
    global snap_data
    with snap_lock:
        snap_data = None
    cli.publish(VISION_CMD, json.dumps({"cmd":"snapshot"}), qos=0)
    t0 = time.time()
    while time.time() - t0 < timeout:
        with snap_lock:
            if snap_data is not None:
                return snap_data
        time.sleep(0.02)
    return None


def latest_frame_bytes():
    with frame_lock:
        if frame_data is None:
            return None
        return frame_data

def preprocess_fullframe(img: Image.Image) -> np.ndarray:
    # Мягкий препроцесс для EasyOCR: серый → автоконтраст → upscale x2
    from PIL import ImageOps
    im = img.convert("L")
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.resize((im.width*2, im.height*2), resample=Image.BICUBIC)
    return np.array(im)

def process_once(client, payload):
    timeout = float(payload.get("timeout", AUTO_TIMEOUT))
    source = "snapshot"
    jpeg = request_snapshot(client, timeout=timeout)
    if jpeg is None:
        source = "frame"
        jpeg = latest_frame_bytes()
        if jpeg is None:
            client.publish(
                OCR_TEXT,
                json.dumps({"ok": False, "error": "no_frame", "timeout": timeout}),
            )
            debug_log("no_frame", timeout=timeout)
            return

    try:
        img = Image.open(io.BytesIO(jpeg))
        if DEBUG_SAVE:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            img.save(DEBUG_DIR / f"easyocr_raw_{ts}.png")
        debug_log("start", source=source)
        np_img = preprocess_fullframe(img)
        # detail=0 → только текст; paragraph=True склеивает блоки
        res = reader.readtext(np_img, detail=0, paragraph=True)
        text = "\n".join([r.strip() for r in res if isinstance(r, str)])
        payload = json.dumps({"ok": True, "text": text})
        client.publish(OCR_TEXT, payload)
        debug_log("publish", chars=len(text.strip()), sample=text[:80])
    except Exception as e:
        client.publish(OCR_TEXT, json.dumps({"ok": False, "error": str(e)}))
        debug_log("error", message=str(e))


def handle_cmd(client, payload):
    if not ready or reader is None:
        client.publish(
            OCR_TEXT,
            json.dumps({
                "ok": False,
                "error": "easyocr_init_failed",
                "detail": globals().get("init_error", ""),
            }),
        )
        return

    cmd = (payload.get("cmd") or "").lower()
    if cmd == "once":
        process_once(client, payload)


def auto_loop(client):
    if not ready or reader is None or AUTO_INTERVAL <= 0:
        return
    while True:
        try:
            process_once(client, {"timeout": AUTO_TIMEOUT})
        except Exception as exc:
            client.publish(
                OCR_TEXT,
                json.dumps({"ok": False, "error": f"auto_loop_failed:{exc}"}),
            )
        time.sleep(AUTO_INTERVAL)


def on_cmd(client, userdata, msg):
    payload = {}
    try:
        payload = json.loads(msg.payload)
    except Exception:
        pass

    threading.Thread(target=handle_cmd, args=(client, dict(payload)), daemon=True).start()


def on_frame(client, userdata, msg):
    global frame_data
    try:
        data = json.loads(msg.payload)
        b64 = data.get("image_b64")
        if not b64:
            return
        raw = base64.b64decode(b64)
        with frame_lock:
            frame_data = raw
    except Exception:
        pass

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(VISION_SNAPSHOT, qos=0)
        client.subscribe(VISION_FRAME, qos=0)
        client.subscribe(OCR_CMD, qos=0)
        client.publish(OCR_TEXT, json.dumps({
            "ok": True,
            "event": "connected",
            "gpu": gpu,
            "listen": [VISION_SNAPSHOT, OCR_CMD]
        }), qos=0)
    else:
        client.publish(OCR_TEXT, json.dumps({"ok": False, "event":"connect_failed","code":int(rc)}), qos=0)

def main():
    cli = mqtt.Client(client_id="ocr_easy", protocol=mqtt.MQTTv311)
    cli.on_connect = on_connect
    cli.message_callback_add(VISION_SNAPSHOT, on_snapshot)
    cli.message_callback_add(VISION_FRAME, on_frame)
    cli.message_callback_add(OCR_CMD, on_cmd)
    cli.connect(MQTT_HOST, MQTT_PORT, 30)
    threading.Thread(target=auto_loop, args=(cli,), daemon=True).start()
    cli.loop_forever()

if __name__ == "__main__":
    main()
