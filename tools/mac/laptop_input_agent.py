import collections
import html
import http.server
import json
import logging
import os
import re
import socketserver
import subprocess
import threading
import time
import urllib.parse
import pyautogui
import paho.mqtt.client as mqtt

# --- clipboard support ---
try:
    import pyperclip
except ImportError:
    pyperclip = None

JETSON_IP = os.environ.get("JETSON_IP", "10.0.0.68")
PORT      = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC     = os.environ.get("ACT_TOPIC", "act/cmd")
CONTROL_TOPIC = os.environ.get("INPUT_CONTROL_TOPIC", "")
CLIENT_ID = "laptop_input"
LOG_LEVEL = os.environ.get("INPUT_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("INPUT_LOG_FILE", "")

ACTION_TTL_MS = float(os.environ.get("INPUT_ACTION_TTL_MS", "500"))
DEDUP_WINDOW_MS = float(os.environ.get("INPUT_DEDUP_WINDOW_MS", "150"))
ACTION_ID_CACHE = int(os.environ.get("INPUT_ACTION_ID_CACHE", "512"))
HEARTBEAT_SEC = float(os.environ.get("INPUT_HEARTBEAT_SEC", "3.0"))
MAX_EVENTS_PER_SEC = int(os.environ.get("INPUT_MAX_EVENTS_PER_SEC", "120"))
MAX_KEYS_HELD = int(os.environ.get("INPUT_MAX_KEYS_HELD", "4"))
IGNORE_RETAINED = os.environ.get("INPUT_IGNORE_RETAINED", "1").strip().lower() not in ("0", "false", "no", "")
PAUSE_FILE = os.environ.get("INPUT_PAUSE_FILE", "/tmp/sg_input_pause")
HTTP_HOST = os.environ.get("INPUT_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("INPUT_HTTP_PORT", "5010"))
HTTP_ENABLED = os.environ.get("INPUT_HTTP_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
INPUT_BOUNDS_CFG = os.environ.get("INPUT_BOUNDS") or os.environ.get("GAME_BOUNDS") or ""
INPUT_BOUNDS_MODE = INPUT_BOUNDS_CFG.strip().lower()
INPUT_BOUNDS_DYNAMIC = INPUT_BOUNDS_MODE in ("window", "front_window", "auto")
BOUNDS_LOCK_FRONT_APP = os.environ.get("INPUT_BOUNDS_LOCK_APP", "1").strip().lower() not in ("0", "false", "no", "")
SCENE_TOPIC = os.environ.get("INPUT_SCENE_TOPIC") or os.environ.get("SCENE_TOPIC", "scene/state")
REQUIRE_GAME = os.environ.get("INPUT_REQUIRE_GAME", "0").strip().lower() not in ("0", "false", "no", "")
SCENE_STALE_SEC = float(os.environ.get("INPUT_SCENE_STALE_SEC", "2.0"))
NO_GAME_PAUSE_SEC = float(os.environ.get("INPUT_NO_GAME_PAUSE_SEC", "2.0"))
DESKTOP_PAUSE_SEC = float(os.environ.get("INPUT_DESKTOP_PAUSE_SEC", "3.0"))
PLAYER_CONF_MIN = float(os.environ.get("INPUT_PLAYER_CONF_MIN", "0.25"))
GAME_KEYWORDS_CFG = os.environ.get("INPUT_GAME_KEYWORDS", "")
DESKTOP_KEYWORDS_CFG = os.environ.get("INPUT_DESKTOP_KEYWORDS", "finder,trash,dock,apple menu,desktop,wallpaper")
FRONT_APP_REQUIRED = os.environ.get("INPUT_REQUIRE_FRONT_APP", "0").strip().lower() not in ("0", "false", "no", "")
FRONT_APP_NAMES_CFG = os.environ.get("INPUT_FRONT_APP") or os.environ.get("INPUT_FRONT_APPS", "")
FRONT_APP_CHECK_SEC = float(os.environ.get("INPUT_FRONT_APP_CHECK_SEC", "1.0"))
FRONT_APP_GRACE_SEC = float(os.environ.get("INPUT_FRONT_APP_GRACE_SEC", "2.0"))
PUBLISH_IDENTITY = os.environ.get("INPUT_PUBLISH_IDENTITY", "0").strip().lower() not in ("0", "false", "no", "")
IDENTITY_TOPIC = os.environ.get("INPUT_IDENTITY_TOPIC", "game/identity")
IDENTITY_PUBLISH_SEC = float(os.environ.get("INPUT_IDENTITY_PUBLISH_SEC", "5.0"))
IDENTITY_SOURCE = os.environ.get("INPUT_IDENTITY_SOURCE", "front_app")

pyautogui.FAILSAFE = False

cli = mqtt.Client(
    client_id=CLIENT_ID,
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    clean_session=True,
)

handlers = [logging.StreamHandler()]
if LOG_FILE:
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE))
    except Exception:
        pass
logging.basicConfig(level=LOG_LEVEL, format="[%(asctime)s] [%(levelname)s] %(message)s", handlers=handlers)
logger = logging.getLogger("laptop_input_agent")

def _parse_bounds(cfg: str):
    if not cfg:
        return None
    if cfg.strip().lower() in ("screen", "fullscreen", "full"):
        width, height = pyautogui.size()
        return (0, 0, width - 1, height - 1)
    try:
        vals = [int(v.strip()) for v in cfg.split(",")]
        if len(vals) != 4:
            raise ValueError
        left, top, right, bottom = vals
        if right <= left or bottom <= top:
            raise ValueError
        return left, top, right, bottom
    except ValueError:
        logger.warning("invalid INPUT_BOUNDS/GAME_BOUNDS=%s, ignoring", cfg)
        return None

INPUT_BOUNDS = None if INPUT_BOUNDS_DYNAMIC else _parse_bounds(INPUT_BOUNDS_CFG)
if INPUT_BOUNDS:
    logger.info("input bounds enabled: %s", INPUT_BOUNDS)
elif INPUT_BOUNDS_DYNAMIC:
    logger.info("input bounds enabled: front_window (auto)")

def _parse_keywords(cfg: str):
    return [item.strip().lower() for item in cfg.split(",") if item.strip()]

GAME_KEYWORDS = _parse_keywords(GAME_KEYWORDS_CFG)
DESKTOP_KEYWORDS = _parse_keywords(DESKTOP_KEYWORDS_CFG)
if REQUIRE_GAME:
    logger.info("scene guard enabled: topic=%s game_keywords=%s desktop_keywords=%s", SCENE_TOPIC, GAME_KEYWORDS, DESKTOP_KEYWORDS)
FRONT_APP_NAMES = _parse_keywords(FRONT_APP_NAMES_CFG)
FRONT_APP_MODE = "strict"
if FRONT_APP_NAMES_CFG.strip().lower() in ("auto", "*", "any"):
    FRONT_APP_MODE = "auto"
if FRONT_APP_REQUIRED:
    if FRONT_APP_MODE == "auto":
        logger.info("front app guard enabled: auto")
    elif not FRONT_APP_NAMES:
        logger.warning("front app guard enabled but INPUT_FRONT_APP is empty")
    else:
        logger.info("front app guard enabled: %s", FRONT_APP_NAMES)
if PUBLISH_IDENTITY:
    logger.info("identity publisher enabled: topic=%s interval=%.1fs", IDENTITY_TOPIC, IDENTITY_PUBLISH_SEC)

state_lock = threading.Lock()
last_msg_ts = 0.0
last_heartbeat_log = 0.0
paused = False
auto_paused = False
auto_pause_reason = None
last_scene_ts = 0.0
last_game_seen_ts = 0.0
last_desktop_seen_ts = 0.0
last_front_app = None
last_front_app_ts = 0.0
last_front_app_ok_ts = 0.0
last_front_app_allowed = None
last_front_app_allowed_ts = 0.0
last_front_app_error_log = 0.0
last_front_window_title = None
last_front_bundle_id = None
last_identity_payload = None
last_identity_publish = 0.0
recent_action_ids = collections.deque()
recent_action_id_set = set()
recent_fingerprints = collections.deque(maxlen=256)
recent_fingerprint_ts = {}
rate_window = collections.deque()
held_keys = set()
held_mouse = set()

def _clamp_point(x: int, y: int):
    if not INPUT_BOUNDS:
        return x, y
    left, top, right, bottom = INPUT_BOUNDS
    return max(left, min(int(x), right)), max(top, min(int(y), bottom))

def _ensure_cursor_in_bounds():
    if not INPUT_BOUNDS:
        return
    cur = pyautogui.position()
    target_x, target_y = _clamp_point(cur.x, cur.y)
    if target_x != cur.x or target_y != cur.y:
        pyautogui.moveTo(target_x, target_y, duration=0)
        logger.debug("mouse clamped: %s %s", target_x, target_y)

def _move_to_clamped(x: int, y: int):
    target_x, target_y = _clamp_point(x, y)
    pyautogui.moveTo(target_x, target_y, duration=0)
    logger.debug("mouse moveTo: %s %s", target_x, target_y)

def _move_rel_clamped(dx: int, dy: int):
    if not INPUT_BOUNDS:
        pyautogui.moveRel(int(dx), int(dy), duration=0)
        logger.debug("mouse moveRel: %s %s", dx, dy)
        return
    cur = pyautogui.position()
    target_x, target_y = _clamp_point(cur.x + int(dx), cur.y + int(dy))
    pyautogui.moveTo(target_x, target_y, duration=0)
    logger.debug("mouse moveRel: %s %s", dx, dy)

def _text_matches(texts, keywords):
    if not keywords:
        return False
    for entry in texts:
        raw = str(entry).lower()
        if not raw:
            continue
        for key in keywords:
            if key in raw:
                return True
    return False

def _scene_texts(state: dict):
    texts = []
    for key in ("text", "texts", "ocr"):
        value = state.get(key)
        if isinstance(value, list):
            texts.extend([str(item) for item in value if item])
        elif isinstance(value, str):
            texts.append(value)
    return texts

def _scene_objects(state: dict):
    objects = []
    for obj in state.get("objects") or []:
        label = obj.get("label") or obj.get("class") or obj.get("name")
        if label:
            objects.append(str(label))
    return objects

def _scene_has_player(state: dict) -> bool:
    player = state.get("player")
    if not isinstance(player, dict):
        return False
    conf = player.get("confidence")
    try:
        return conf is None or float(conf) >= PLAYER_CONF_MIN
    except (TypeError, ValueError):
        return False

def _handle_scene(state: dict):
    global last_scene_ts, last_game_seen_ts, last_desktop_seen_ts
    now = time.time()
    with state_lock:
        last_scene_ts = now
    texts = _scene_texts(state)
    objects = _scene_objects(state)
    if DESKTOP_KEYWORDS and (_text_matches(texts, DESKTOP_KEYWORDS) or _text_matches(objects, DESKTOP_KEYWORDS)):
        with state_lock:
            last_desktop_seen_ts = now
    game_signal = _scene_has_player(state)
    if GAME_KEYWORDS and (_text_matches(texts, GAME_KEYWORDS) or _text_matches(objects, GAME_KEYWORDS)):
        game_signal = True
    if game_signal:
        with state_lock:
            last_game_seen_ts = now

def _front_app_matches(name: str) -> bool:
    if FRONT_APP_MODE == "auto":
        if not last_front_app_allowed:
            return False
        return last_front_app_allowed in name.lower()
    if not FRONT_APP_NAMES:
        return False
    lowered = name.lower()
    return any(token in lowered for token in FRONT_APP_NAMES)

def _set_front_app_allowed(name: str, reason: str):
    global last_front_app_allowed, last_front_app_allowed_ts
    if not name:
        return
    lowered = name.lower()
    if last_front_app_allowed == lowered:
        return
    last_front_app_allowed = lowered
    last_front_app_allowed_ts = time.time()
    logger.info("front app allowed set to '%s' (%s)", name, reason)

def _maybe_update_front_app_allowed(name: str):
    if FRONT_APP_MODE != "auto":
        return
    if not last_front_app_allowed and name:
        _set_front_app_allowed(name, "auto_first_seen")

def _run_osascript(script: str):
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err or f"osascript exit {result.returncode}")
    return result.stdout.strip() or None

def _get_front_app_name():
    return _run_osascript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
    )

def _get_front_window_title():
    script = (
        'tell application "System Events"\n'
        '  tell (first application process whose frontmost is true)\n'
        '    if (count of windows) > 0 then\n'
        '      return name of front window\n'
        '    end if\n'
        '  end tell\n'
        'end tell'
    )
    return _run_osascript(script)

def _get_front_bundle_id():
    return _run_osascript(
        'tell application "System Events" to get bundle identifier of first application process whose frontmost is true'
    )

def _get_front_window_bounds():
    script = (
        'tell application "System Events"\n'
        '  tell (first application process whose frontmost is true)\n'
        '    if (count of windows) > 0 then\n'
        '      set pos to position of front window\n'
        '      set sz to size of front window\n'
        '      return (item 1 of pos) & "," & (item 2 of pos) & "," & (item 1 of sz) & "," & (item 2 of sz)\n'
        '    end if\n'
        '  end tell\n'
        'end tell'
    )
    raw = _run_osascript(script)
    if not raw:
        return None
    parts = [int(p) for p in re.findall(r"-?\d+", raw)]
    if len(parts) < 4:
        return None
    left, top, width, height = parts
    if width <= 0 or height <= 0:
        return None
    right = left + width - 1
    bottom = top + height - 1
    return (left, top, right, bottom)

def _maybe_update_bounds_from_front_window(name: str):
    global INPUT_BOUNDS
    if not INPUT_BOUNDS_DYNAMIC:
        return
    if FRONT_APP_REQUIRED and not name:
        return
    if FRONT_APP_REQUIRED and name and not _front_app_matches(name):
        return
    if BOUNDS_LOCK_FRONT_APP:
        if not name:
            return
        if last_front_app_allowed and not _front_app_matches(name):
            return
    try:
        bounds = _get_front_window_bounds()
    except Exception as exc:
        now = time.time()
        global last_front_app_error_log
        if now - last_front_app_error_log > 10:
            last_front_app_error_log = now
            logger.warning("front window bounds check failed: %s", exc)
        return
    if not bounds:
        return
    if bounds != INPUT_BOUNDS:
        INPUT_BOUNDS = bounds
        logger.info("input bounds updated: %s", INPUT_BOUNDS)

def _slugify(value: str):
    if not value:
        return "unknown_game"
    lowered = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "unknown_game"

def _maybe_publish_identity(now: float, name: str, title: str, bundle: str):
    global last_identity_payload, last_identity_publish
    if not PUBLISH_IDENTITY or not IDENTITY_TOPIC:
        return
    raw = name or title or bundle or ""
    payload = {
        "ok": True,
        "timestamp": now,
        "app_name": name,
        "window_title": title,
        "bundle_id": bundle,
        "game_id": _slugify(raw) if raw else "unknown_game",
        "source": IDENTITY_SOURCE,
    }
    if payload == last_identity_payload and (now - last_identity_publish) < IDENTITY_PUBLISH_SEC:
        return
    cli.publish(IDENTITY_TOPIC, json.dumps(payload))
    last_identity_payload = payload
    last_identity_publish = now

def _front_app_loop():
    global last_front_app, last_front_app_ts, last_front_app_ok_ts, last_front_app_error_log
    global last_front_window_title, last_front_bundle_id
    if not FRONT_APP_REQUIRED and not PUBLISH_IDENTITY and not INPUT_BOUNDS_DYNAMIC:
        return
    sleep_sec = max(0.2, FRONT_APP_CHECK_SEC)
    while True:
        time.sleep(sleep_sec)
        name = None
        title = None
        bundle = None
        try:
            name = _get_front_app_name()
            if PUBLISH_IDENTITY:
                title = _get_front_window_title()
                bundle = _get_front_bundle_id()
        except Exception as exc:
            now = time.time()
            if now - last_front_app_error_log > 10:
                last_front_app_error_log = now
                logger.warning("front app check failed: %s", exc)
        now = time.time()
        with state_lock:
            last_front_app = name
            last_front_app_ts = now
            _maybe_update_front_app_allowed(name)
            if name and _front_app_matches(name):
                last_front_app_ok_ts = now
            if PUBLISH_IDENTITY:
                last_front_window_title = title
                last_front_bundle_id = bundle
        if PUBLISH_IDENTITY:
            try:
                _maybe_publish_identity(now, name, title, bundle)
            except Exception as exc:
                if now - last_front_app_error_log > 10:
                    last_front_app_error_log = now
                    logger.warning("identity publish failed: %s", exc)
        _maybe_update_bounds_from_front_window(name)

def _update_auto_pause():
    global auto_paused, auto_pause_reason
    if not REQUIRE_GAME and not FRONT_APP_REQUIRED:
        if auto_paused:
            auto_paused = False
            auto_pause_reason = None
            logger.info("auto-resumed: guard disabled")
        return
    now = time.time()
    with state_lock:
        scene_ts = last_scene_ts
        game_ts = last_game_seen_ts
        desktop_ts = last_desktop_seen_ts
        front_ok_ts = last_front_app_ok_ts
        front_name = last_front_app
    reason = None
    if FRONT_APP_REQUIRED:
        if FRONT_APP_MODE != "auto" and not FRONT_APP_NAMES:
            reason = "front_app_not_configured"
        elif FRONT_APP_MODE == "auto" and not last_front_app_allowed:
            reason = "front_app_unset"
        elif front_ok_ts and (now - front_ok_ts) <= FRONT_APP_GRACE_SEC:
            reason = None
        else:
            reason = "front_app_mismatch" if front_name else "front_app_unknown"
    if reason is None and REQUIRE_GAME:
        if scene_ts and SCENE_STALE_SEC > 0 and (now - scene_ts) > SCENE_STALE_SEC:
            reason = "no_scene"
        elif desktop_ts and DESKTOP_PAUSE_SEC > 0 and (now - desktop_ts) < DESKTOP_PAUSE_SEC:
            reason = "desktop_detected"
        elif game_ts and (now - game_ts) <= NO_GAME_PAUSE_SEC:
            reason = None
        else:
            reason = "no_game_signal"
    if reason:
        if not auto_paused or auto_pause_reason != reason:
            auto_paused = True
            auto_pause_reason = reason
            _panic_release()
            logger.warning("auto-paused: %s", reason)
    else:
        if auto_paused:
            auto_paused = False
            auto_pause_reason = None
            logger.info("auto-resumed: game visible")

def on_connect(client, userdata, flags, reason_code, properties=None):
    topics = [(TOPIC, 1)]
    if CONTROL_TOPIC:
        topics.append((CONTROL_TOPIC, 1))
    if SCENE_TOPIC:
        topics.append((SCENE_TOPIC, 0))
    for topic, qos in topics:
        client.subscribe(topic, qos=qos)
    logger.info("connected to %s:%s rc=%s topics=%s", JETSON_IP, PORT, getattr(reason_code, "value", reason_code), [t for t, _ in topics])

def on_message(client, userdata, msg):
    global last_msg_ts
    try:
        if IGNORE_RETAINED and getattr(msg, "retain", False):
            return
        payload = msg.payload.decode("utf-8", "ignore")
        data = json.loads(payload) if payload.startswith("{") else {"key": payload}
        if not isinstance(data, dict):
            return
        if msg.topic == SCENE_TOPIC:
            _handle_scene(data)
            return
        now = time.time()
        with state_lock:
            last_msg_ts = now

        if msg.topic == CONTROL_TOPIC:
            _handle_control(data)
            return

        if _is_paused():
            return

        action = str(data.get("action") or data.get("act") or data.get("label") or "").lower().strip()
        if not action and "key" in data:
            action = "key_press"

        if action == "panic":
            _panic_release()
            return

        ts = _coerce_ts(data.get("timestamp") or data.get("ts"))
        if ts and ACTION_TTL_MS > 0 and (now - ts) * 1000.0 > ACTION_TTL_MS:
            return

        action_id = data.get("action_id") or data.get("id") or data.get("uuid")
        if action_id and _seen_action_id(str(action_id)):
            return
        if not action_id and DEDUP_WINDOW_MS > 0:
            fingerprint = _fingerprint(action, data)
            if _seen_fingerprint(fingerprint, now):
                return

        if _rate_limited(now):
            return

        if action in {"click_primary", "click_secondary", "click_middle"}:
            button = "left" if action == "click_primary" else "right" if action == "click_secondary" else "middle"
            _ensure_cursor_in_bounds()
            pyautogui.click(button=button); logger.info("mouse click: %s", button); return
        if action in {"mouse_click", "mouse_down", "mouse_up"}:
            button = data.get("button", "left")
            if action == "mouse_click":
                _ensure_cursor_in_bounds()
                pyautogui.click(button=button); logger.info("mouse click: %s", button); return
            if action == "mouse_down":
                _ensure_cursor_in_bounds()
                pyautogui.mouseDown(button=button); held_mouse.add(button); logger.info("mouse down: %s", button); return
            pyautogui.mouseUp(button=button); held_mouse.discard(button); logger.info("mouse up: %s", button); return
        if action == "mouse_move":
            if "dx" in data and "dy" in data:
                _move_rel_clamped(int(data["dx"]), int(data["dy"]))
            elif "x" in data and "y" in data:
                _move_to_clamped(int(data["x"]), int(data["y"]))
            return
        if action == "mouse_hold":
            button = data.get("button", "left")
            _ensure_cursor_in_bounds()
            pyautogui.mouseDown(button=button); held_mouse.add(button); logger.info("mouse down: %s", button); return
        if action == "mouse_release":
            button = data.get("button", "left")
            pyautogui.mouseUp(button=button); held_mouse.discard(button); logger.info("mouse up: %s", button); return

        if action in {"key_press", "key_down", "key_up", "key"}:
            key = data.get("key")
            if not key:
                return
            if action == "key_down":
                if len(held_keys) >= MAX_KEYS_HELD:
                    return
                pyautogui.keyDown(key); held_keys.add(key); logger.info("down: %s", key); return
            if action == "key_up":
                pyautogui.keyUp(key); held_keys.discard(key); logger.info("up: %s", key); return
            pyautogui.press(key); logger.info("pressed: %s", key); return

        if action == "scroll":
            dx = int(data.get("axis1", 0))
            dy = int(data.get("axis2", 0))
            if dx or dy:
                pyautogui.scroll(dy)
                logger.info("scroll: %s", dy)
            return

        if action == "type":
            txt = str(data.get("text",""))
            interval = float(data.get("interval", 0.02))
            pyautogui.typewrite(txt, interval=interval)
            logger.info("type: %s chars", len(txt))
            return

        if action == "paste":
            txt = str(data.get("text",""))
            if not pyperclip:
                logger.warning("paste: pyperclip not installed"); return
            pyperclip.copy(txt)
            pyautogui.hotkey("command","v")
            logger.info("paste: %s chars", len(txt))
            return

        if action == "hotkey":
            keys = data.get("keys", [])
            if keys:
                pyautogui.hotkey(*keys)
                logger.info("hotkey: %s", "+".join(keys))
            return

    except Exception as e:
        logger.error("error: %s", e)

def on_disconnect(client, userdata, reason_code=None, properties=None, *args):
    # Paho MQTT v2 passes (disconnect_flags, reason_code, properties)
    if args:
        reason_code = properties
        properties = args[0]
    logger.warning("disconnected: %s", reason_code)
    delay = 1.0
    while True:
        try:
            time.sleep(delay)
            client.reconnect(); logger.info("reconnected"); return
        except Exception as e:
            logger.warning("reconnect error: %s", e); delay = min(delay*2, 30)

def _coerce_ts(ts):
    if ts is None:
        return None
    try:
        ts = float(ts)
    except Exception:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return ts

def _seen_action_id(action_id: str) -> bool:
    if not action_id:
        return False
    if action_id in recent_action_id_set:
        return True
    recent_action_id_set.add(action_id)
    recent_action_ids.append(action_id)
    if ACTION_ID_CACHE > 0 and len(recent_action_ids) > ACTION_ID_CACHE:
        old = recent_action_ids.popleft()
        recent_action_id_set.discard(old)
    return False

def _fingerprint(action: str, data: dict) -> str:
    return f"{action}|{data.get('key')}|{data.get('dx')}|{data.get('dy')}|{data.get('button')}|{data.get('target_px')}"

def _seen_fingerprint(fp: str, now: float) -> bool:
    cutoff = now - (DEDUP_WINDOW_MS / 1000.0)
    last = recent_fingerprint_ts.get(fp)
    if last and last >= cutoff:
        return True
    recent_fingerprint_ts[fp] = now
    recent_fingerprints.append((fp, now))
    while recent_fingerprints and recent_fingerprints[0][1] < cutoff:
        old_fp, _ = recent_fingerprints.popleft()
        if recent_fingerprint_ts.get(old_fp, 0) < cutoff:
            recent_fingerprint_ts.pop(old_fp, None)
    return False

def _rate_limited(now: float) -> bool:
    if MAX_EVENTS_PER_SEC <= 0:
        return False
    rate_window.append(now)
    cutoff = now - 1.0
    while rate_window and rate_window[0] < cutoff:
        rate_window.popleft()
    return len(rate_window) > MAX_EVENTS_PER_SEC

def _panic_release():
    for key in list(held_keys):
        try:
            pyautogui.keyUp(key)
        except Exception:
            pass
    held_keys.clear()
    for button in list(held_mouse):
        try:
            pyautogui.mouseUp(button=button)
        except Exception:
            pass
    held_mouse.clear()
    logger.warning("panic: released held inputs")

def _set_paused(value: bool):
    global paused
    if paused == value:
        return
    paused = value
    if PAUSE_FILE:
        try:
            if value:
                with open(PAUSE_FILE, "w", encoding="utf-8") as f:
                    f.write("paused\n")
            else:
                if os.path.exists(PAUSE_FILE):
                    os.remove(PAUSE_FILE)
        except Exception:
            pass
    if value:
        _panic_release()
        logger.warning("paused input bridge")
    else:
        logger.info("resumed input bridge")

def _is_paused() -> bool:
    if PAUSE_FILE and os.path.exists(PAUSE_FILE):
        _set_paused(True)
        return True
    _update_auto_pause()
    if auto_paused:
        return True
    return paused

def _handle_control(data: dict):
    cmd = str(data.get("cmd") or data.get("action") or "").lower().strip()
    if cmd == "pause":
        _set_paused(True); return
    if cmd == "resume":
        _set_paused(False); return
    if cmd == "panic":
        _panic_release(); return

def _status_payload() -> dict:
    now = time.time()
    with state_lock:
        last_seen = last_msg_ts
        scene_ts = last_scene_ts
        game_ts = last_game_seen_ts
        front_name = last_front_app
        front_ok_ts = last_front_app_ok_ts
        front_ts = last_front_app_ts
        front_allowed = last_front_app_allowed
        front_allowed_ts = last_front_app_allowed_ts
    return {
        "paused": _is_paused(),
        "auto_paused": auto_paused,
        "auto_pause_reason": auto_pause_reason,
        "last_msg_ts": last_seen or None,
        "last_msg_age_sec": round(now - last_seen, 3) if last_seen else None,
        "last_scene_age_sec": round(now - scene_ts, 3) if scene_ts else None,
        "last_game_seen_sec": round(now - game_ts, 3) if game_ts else None,
        "front_app_required": FRONT_APP_REQUIRED,
        "front_app_mode": FRONT_APP_MODE,
        "front_app": front_name,
        "front_app_age_sec": round(now - front_ts, 3) if front_ts else None,
        "front_app_ok_age_sec": round(now - front_ok_ts, 3) if front_ok_ts else None,
        "front_app_allowed": front_allowed,
        "front_app_allowed_age_sec": round(now - front_allowed_ts, 3) if front_allowed_ts else None,
        "input_bounds": INPUT_BOUNDS,
        "input_bounds_dynamic": INPUT_BOUNDS_DYNAMIC,
        "input_bounds_lock_app": BOUNDS_LOCK_FRONT_APP,
        "topic": TOPIC,
        "control_topic": CONTROL_TOPIC or None,
        "scene_topic": SCENE_TOPIC or None,
        "jetson": f"{JETSON_IP}:{PORT}",
        "max_events_per_sec": MAX_EVENTS_PER_SEC,
        "max_keys_held": MAX_KEYS_HELD,
    }

def _render_index(status: dict) -> str:
    status_json = html.escape(json.dumps(status, indent=2))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>MQTT Input Bridge</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #111; color: #eee; }}
      button {{ margin: 6px; padding: 12px 24px; font-size: 15px; border: none; border-radius: 6px; cursor: pointer; }}
      .danger {{ background: #d9534f; color: #fff; }}
      .success {{ background: #5cb85c; color: #fff; }}
      .info {{ background: #0275d8; color: #fff; }}
      pre {{ background: #222; padding: 12px; border-radius: 6px; }}
      form.inline {{ display: inline-flex; }}
    </style>
  </head>
  <body>
    <h1>MQTT Input Bridge</h1>
    <div class="controls">
      <form class="inline" action="/pause" method="post">
        <button class="danger" type="submit">Pause</button>
      </form>
      <form class="inline" action="/resume" method="post">
        <button class="success" type="submit">Resume</button>
      </form>
      <form class="inline" action="/panic" method="post">
        <button class="info" type="submit">Panic Release</button>
      </form>
      <button class="info" onclick="window.location.reload()">Refresh</button>
    </div>
    <h3>Status</h3>
    <pre>{status_json}</pre>
  </body>
</html>
"""

class _ControlHandler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, content_type: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def _handle_action(self, action: str):
        if action == "pause":
            _set_paused(True)
        elif action == "resume":
            _set_paused(False)
        elif action == "panic":
            _panic_release()
        elif action == "set_front_app":
            try:
                name = _get_front_app_name()
            except Exception as exc:
                logger.warning("front app set failed: %s", exc)
                return
            if name:
                _set_front_app_allowed(name, "manual")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, _render_index(_status_payload()))
            return
        if path == "/status":
            self._send(200, json.dumps(_status_payload()), "application/json")
            return
        if path in ("/pause", "/resume", "/panic"):
            self._handle_action(path.lstrip("/"))
            self._send(200, _render_index(_status_payload()))
            return
        if path == "/set_front_app":
            self._handle_action("set_front_app")
            self._send(200, _render_index(_status_payload()))
            return
        self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/pause", "/resume", "/panic"):
            self._handle_action(path.lstrip("/"))
            self._send(200, _render_index(_status_payload()))
            return
        if path == "/set_front_app":
            self._handle_action("set_front_app")
            self._send(200, _render_index(_status_payload()))
            return
        if path == "/status":
            self._send(200, json.dumps(_status_payload()), "application/json")
            return
        self._send(404, "not found", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        logger.debug("http %s - %s", self.client_address[0], fmt % args)

def _start_http_server():
    if not HTTP_ENABLED or HTTP_PORT <= 0:
        return
    try:
        server = socketserver.ThreadingTCPServer((HTTP_HOST, HTTP_PORT), _ControlHandler)
    except OSError as exc:
        logger.warning("HTTP control failed to start on %s:%s: %s", HTTP_HOST, HTTP_PORT, exc)
        return
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("HTTP control listening on http://%s:%s", HTTP_HOST, HTTP_PORT)

def _heartbeat_loop():
    global last_heartbeat_log
    while True:
        time.sleep(max(0.5, HEARTBEAT_SEC / 2.0))
        now = time.time()
        with state_lock:
            last_seen = last_msg_ts
        if last_seen <= 0:
            continue
        if now - last_seen > HEARTBEAT_SEC and now - last_heartbeat_log > HEARTBEAT_SEC:
            last_heartbeat_log = now
            logger.warning("no commands for %.1fs (topic=%s)", now - last_seen, TOPIC)

threading.Thread(target=_heartbeat_loop, daemon=True).start()
threading.Thread(target=_front_app_loop, daemon=True).start()
_start_http_server()

cli.on_connect = on_connect
cli.on_message = on_message
cli.on_disconnect = on_disconnect
cli.connect(JETSON_IP, PORT, 60)
cli.loop_forever()
