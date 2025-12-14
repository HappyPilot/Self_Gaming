#!/usr/bin/env python3
import os, json, time, base64, io
import paho.mqtt.client as mqtt

# OCR
from PIL import Image
import pytesseract

# --- ENV/топики ---
MQTT_HOST = os.environ.get("MQTT_HOST", "mq")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
# куда шлём команду vision'у сделать снимок:
VISION_CMD = os.environ.get("VISION_CMD", "vision/cmd")
# где слушаем ответ со снапшотом:
VISION_SNAPSHOT = os.environ.get("VISION_SNAPSHOT", "vision/snapshot")
# куда публикуем текст:
OCR_TEXT = os.environ.get("OCR_TEXT", "ocr/text")
# команда для единичного запуска:
OCR_CMD = os.environ.get("OCR_CMD", "ocr/cmd")

# языки tesseract (можно 'eng', 'rus' или 'eng+rus')
OCR_LANGS = os.environ.get("OCR_LANGS", "eng+rus")

# простая задержка между попытками
REQUEST_TIMEOUT = float(os.environ.get("OCR_REQ_TIMEOUT", "2.0"))

state = {"last_b64": None}

def on_connect(cli, ud, flags, rc, props=None):
    cli.subscribe(VISION_SNAPSHOT)
    cli.subscribe(OCR_CMD)
    print(json.dumps({"ok": True, "event": "connected", "listen": [VISION_SNAPSHOT, OCR_CMD]}), flush=True)

def on_message(cli, ud, msg):
    topic = msg.topic
    payload = msg.payload.decode("utf-8", "ignore")
    if topic == VISION_SNAPSHOT:
        try:
            data = json.loads(payload)
        except Exception:
            data = {}
        if data.get("ok") and "image_b64" in data:
            state["last_b64"] = data["image_b64"]
    elif topic == OCR_CMD:
        # ожидаем JSON вида {"cmd":"once"} или простую строку "once"
        cmd = None
        try:
            data = json.loads(payload)
            cmd = (data.get("cmd") or "").lower()
        except Exception:
            cmd = payload.strip().lower()
        if cmd in ("once","run","start"):
            ocr_once(cli)

def ocr_once(cli: mqtt.Client):
    # 1) запросить снапшот у vision
    cli.publish(VISION_CMD, json.dumps({"cmd": "snapshot"}), qos=0)
    # 2) дождаться ответа или таймаут
    t0 = time.time()
    got = None
    while time.time() - t0 < REQUEST_TIMEOUT:
        if state.get("last_b64"):
            got = state.pop("last_b64")
            break
        time.sleep(0.05)
    if not got:
        cli.publish(OCR_TEXT, json.dumps({"ok": False, "error": "no_snapshot"}))
        return

    # 3) декодировать JPEG и прогнать Tesseract
    try:
        img = Image.open(io.BytesIO(base64.b64decode(got)))
        # Грейскейл + лёгкий апскейл для OCR (без кропа — читаем весь кадр)
        up = img.convert("L").resize((img.width * 2, img.height * 2))
        text = pytesseract.image_to_string(
            up,
            lang=OCR_LANGS,
            config="--psm 6 --oem 1 -c preserve_interword_spaces=1 -c user_defined_dpi=200",
        )
        out = {"ok": True, "text": text}
    except Exception as e:
        out = {"ok": False, "error": str(e)}

    # 4) опубликовать результат
    cli.publish(OCR_TEXT, json.dumps(out), qos=0)

def main():
    cli = mqtt.Client(client_id="ocr_agent",protocol=mqtt.MQTTv311)
    cli.on_connect = on_connect
    cli.on_message = on_message
    cli.connect(MQTT_HOST, MQTT_PORT, 30)
    cli.loop_forever()

if __name__ == "__main__":
    main()
