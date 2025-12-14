# Self_Gaming — Master Codex Task Pack
(DeepStream V4L2 + Universal OCR optimization + Policy Brain shadow-mode)

## Contents
1. DeepStream: V4L2 as real video source (HTTP optional viewer)
2. OCR: Universal low-CPU “radar” mode + frame-change gating
3. Policy Brain: shadow agent that vectorizes scene/state → dummy model → policy_brain/cmd + compares vs act/cmd
4. Definition of Done

---

## 1) DeepStream V4L2 task (HTTP optional)
# Codex Task: Make DeepStream use V4L2 (USB-HDMI capture) as the real video source; keep HTTP only for UI viewing

## Goal
1) `perception_ds` (DeepStream) must **NOT** depend on `http://*:5008/video_feed` anymore.
2) DeepStream must read frames directly from **V4L2** (e.g. `/dev/video0` from USB-HDMI capture).
3) HTTP streaming (port 5008) becomes **optional** and used only for humans/UI (“I want to see what the agent sees”).
4) After implementing, generate a **detailed project overview** file describing what’s in the repo and how it works end-to-end, so ChatGPT can read and analyze it.

---

## Why this is needed (current failure)
Right now `perception_ds` is restarting because `souphttpsrc` cannot connect:
```
Could not connect: Connection refused (4), URL: http://10.0.0.68:5008/video_feed
```
So DeepStream never stabilizes and GPU stays idle (`GR3D_FREQ 0%`).

---

## Step 0 — Find the right V4L2 device (host)
On Jetson host, identify the USB capture device node.

Run:
```bash
ls -l /dev/v4l/by-id || true
ls -l /dev/video* || true
# Optional if installed:
v4l2-ctl --list-devices || true
```

Pick the device that corresponds to the HDMI capture (example: `/dev/video0`).

We will parameterize it via env: `DS_V4L2_DEVICE=/dev/video0`.

---

## Step 1 — Patch docker-compose to pass /dev/videoX into perception_ds
Edit `docker-compose.yml` (or compose file where `perception_ds` is defined).

### Required changes for `perception_ds` service
- Add `devices` mapping for the V4L2 device
- Add `/dev/v4l` mapping (optional but helps if multiple devices)
- Add `/run/udev` read-only (optional, helps some GStreamer device discovery)
- Ensure container can access video group (best-effort)

#### Patch (example)
```yaml
services:
  perception_ds:
    # ... existing fields ...
    devices:
      - "${DS_V4L2_DEVICE:-/dev/video0}:${DS_V4L2_DEVICE:-/dev/video0}"
    volumes:
      - /dev/v4l:/dev/v4l
      - /run/udev:/run/udev:ro
    environment:
      - DS_SOURCE=v4l2
      - DS_V4L2_DEVICE=${DS_V4L2_DEVICE:-/dev/video0}
      - DS_V4L2_WIDTH=${DS_V4L2_WIDTH:-1280}
      - DS_V4L2_HEIGHT=${DS_V4L2_HEIGHT:-720}
      - DS_V4L2_FPS=${DS_V4L2_FPS:-30}
      - DS_V4L2_FORMAT=${DS_V4L2_FORMAT:-MJPG}   # MJPG or YUYV; depends on device
```

Notes:
- Some capture cards output MJPG; others output YUYV. We’ll support both.
- If your compose already defines environment keys, merge them; don’t overwrite.

---

## Step 2 — Modify DeepStream pipeline to use v4l2src instead of souphttpsrc
Locate where the source element is created. From logs, it’s likely in:
- `/app/deepstream_mqtt.py` (or similar)
- and/or configs under `/app/docker/deepstream/*`

### Strategy
Implement a switch:
- If `DS_SOURCE=v4l2`, build pipeline using `v4l2src device=$DS_V4L2_DEVICE`
- Else (default), keep old behavior (`souphttpsrc location=$HTTP_URL`) for debugging/back-compat.

### Implementation details (GStreamer)
Typical safe source chain for USB capture:

**MJPG:**
```
v4l2src device=/dev/video0 !
image/jpeg, width=1280, height=720, framerate=30/1 !
jpegparse ! nvv4l2decoder !
nvvidconv ! video/x-raw(memory:NVMM), format=NV12 !
nvstreammux ...
```

**YUYV:**
```
v4l2src device=/dev/video0 !
video/x-raw, format=YUY2, width=1280, height=720, framerate=30/1 !
videoconvert !
nvvideoconvert ! video/x-raw(memory:NVMM), format=NV12 !
nvstreammux ...
```

Your pipeline already has `nvstreammux` + `nvinfer` + MQTT publishing; keep that.

### Patch requirements
1) Add env parsing in `deepstream_mqtt.py`:
   - `DS_SOURCE` default `http`
   - `DS_V4L2_DEVICE`, `DS_V4L2_WIDTH`, `DS_V4L2_HEIGHT`, `DS_V4L2_FPS`, `DS_V4L2_FORMAT`
2) When building the pipeline, choose source branch.
3) Log the chosen source and caps so future debugging is easy.

---

## Step 3 — Keep HTTP “viewer” separate (optional)
Port 5008 is useful for humans, but it must not be a hard dependency.

Make sure:
- `http_bridge` and/or `vision_agent` can still run and expose `:5008` if desired.
- But `perception_ds` should not care if `:5008` is down.

Optional improvement:
- Add compose profile `debug` for HTTP viewer containers.
- Example:
```yaml
services:
  http_bridge:
    profiles: ["debug"]
  pyds_debug:
    profiles: ["debug"]
```

Then:
```bash
docker compose --profile debug up -d
```

---

## Step 4 — Validation checklist (must pass)
After patch:

### A) DeepStream should stop restarting
```bash
docker compose ps perception_ds
docker logs --tail 80 perception_ds
```
Expected: no “connection refused”; pipeline stays running.

### B) GPU should show activity when inference runs
```bash
timeout 10s tegrastats
# Look for GR3D_FREQ > 0% while DS is running inference
```

### C) Confirm the pipeline is ingesting frames
Add one log line on each `pad-added` / or periodic FPS print if you have it.
At minimum, ensure no warnings about caps negotiation failing.

### D) Confirm object messages still publish to MQTT (if applicable)
If DS publishes detection results to MQTT topic(s), run:
```bash
docker exec -it mq mosquitto_sub -t 'vision/#' -v -W 5 || true
docker exec -it mq mosquitto_sub -t 'objects/#' -v -W 5 || true
```
(adjust topics to your project’s actual topic names)

---

## Step 5 — Add a “Project Overview Generator” and run it at the end
Create: `tools/generate_project_overview.py`

### What it must do
Write a Markdown file: `PROJECT_OVERVIEW.md` containing:
1) Repo summary (purpose, top-level dirs, what problem it solves)
2) Runtime architecture:
   - list docker compose services
   - what each service does (agents)
   - main data flow (video → vision → OCR → policy → act)
3) MQTT topic map:
   - topics published/subscribed by key agents (best-effort by scanning code)
4) Configuration:
   - key env vars (from compose + `.env` if exists)
   - “modes” (debug vs prod, http viewer on/off, v4l2 source on/off)
5) Current health snapshot instructions:
   - the exact commands you used to check load/mem/docker stats/tegrastats
6) Known pain points + troubleshooting:
   - V4L2 device mismatch
   - caps negotiation failures
   - DS restart loops
   - CPU overload
   - OCR load and recommended defaults

### How it should gather info (no external deps)
Use Python stdlib:
- `subprocess` to run:
  - `git rev-parse --short HEAD` (if git present)
  - `docker compose ps`
  - `docker compose config` (save raw output)
- filesystem scan:
  - list top-level folders
  - grep-like scan for MQTT topic strings in `.py` files
  - detect agent entrypoints by searching for `python3 /app/*.py` or `CMD` in Dockerfiles

### Minimal code skeleton (Codex should implement fully)
```python
#!/usr/bin/env python3
import os, re, subprocess, textwrap, pathlib, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "PROJECT_OVERVIEW.md"

def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return f"[command failed] {cmd}\\n{e}\\n"

def scan_topics():
    topics = set()
    for p in ROOT.rglob("*.py"):
        try:
            s = p.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.findall(r"['\\\"]([a-zA-Z0-9_\\-]+/[a-zA-Z0-9_\\-/#]+)['\\\"]", s):
            if "/" in m and len(m) <= 120:
                topics.add(m)
    return sorted(topics)

def main():
    now = datetime.datetime.now().isoformat(timespec="seconds")
    top_dirs = sorted([x.name for x in ROOT.iterdir() if x.is_dir() and not x.name.startswith(".")])

    git_rev = sh(["bash", "-lc", "git rev-parse --short HEAD || true"]).strip()
    compose_ps = sh(["bash", "-lc", "docker compose ps || true"])
    compose_cfg = sh(["bash", "-lc", "docker compose config || true"])

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "compose_config.yaml").write_text(compose_cfg)

    topics = scan_topics()

    OUT.write_text(textwrap.dedent(f\"\"\"\
    # Project Overview (auto-generated)
    Generated: {now}
    Git: {git_rev}

    ## What this repo is
    (Fill in: short description)

    ## Top-level directories
    {chr(10).join(f"- `{d}/`" for d in top_dirs)}

    ## Docker Compose services (current)
    ```text
    {compose_ps.strip()}
    ```

    ## Compose config snapshot
    Saved to: `docs/compose_config.yaml`

    ## MQTT topics discovered (best-effort scan)
    {chr(10).join(f"- `{t}`" for t in topics)}

    ## Data flow (expected)
    (Fill in: Video -> Vision -> OCR -> Policy -> Act)

    ## Modes
    - `DS_SOURCE=v4l2` uses `/dev/videoX` direct capture
    - HTTP :5008 is optional (viewer/debug)

    ## Troubleshooting
    (Fill in: common failure modes + commands)
    \"\"\"))
    print(f\"Wrote {OUT}\")

if __name__ == "__main__":
    main()
```

### End-of-task requirement
After all patches, run:
```bash
python3 tools/generate_project_overview.py
```
and ensure `PROJECT_OVERVIEW.md` is created and readable.

---

## Deliverables (must be produced)
1) Compose patch: `perception_ds` can access `/dev/videoX` and has DS_* envs.
2) DeepStream source switch: `DS_SOURCE=v4l2` works; `DS_SOURCE=http` still possible.
3) `perception_ds` no longer restarts because of HTTP.
4) `tools/generate_project_overview.py`
5) `PROJECT_OVERVIEW.md` generated at end.

---

## Quick “golden” run commands (after implementation)
```bash
# set capture device
export DS_V4L2_DEVICE=/dev/video0
export DS_SOURCE=v4l2

# restart DS
docker compose up -d --force-recreate perception_ds

# check it stays up
docker compose ps perception_ds
docker logs --tail 80 perception_ds

# check GPU tick
timeout 10s tegrastats
```

---

## 2) Universal OCR mode patch (Jetson)
# Universal OCR mode — Codex-ready patch (Jetson)

## Current finding (from your logs)
`PID 961126` is `python3 /app/ocr_easy_agent.py` inside container `ocr_easy_agent` (`CID=5592834d...`).  
Env confirms it is *already* in “radar-ish” settings (interval=4s, max width=640, max results=10), but it still burns CPU (~158%).

Key suspects that commonly keep PaddleOCR hot even in radar mode:
- **Paddle/OpenMP thread fan-out** (default can use many threads → >100% CPU)
- **Debug saving** (`OCR_DEBUG_SAVE=1`) writing images frequently
- **Two-language decoding** (`OCR_LANGS=en,ru`) adds work
- **Always-OCR every interval** even when the frame didn’t change much

## Goal
Make OCR stable for **any game** by:
1) keeping a cheap always-on OCR “radar”,
2) running expensive OCR only when the scene/text *changes* or when UI regions demand it,
3) ensuring heartbeat messages to avoid timeouts.

---

## Patch 1 — Compose: hard cap CPU threads + disable debug saving (safe, highest ROI)

### Edit: `docker-compose.yml` (service: `ocr_easy_agent`)
Add/replace these environment variables:

```yaml
services:
  ocr_easy_agent:
    environment:
      # Keep radar settings (tune as needed)
      - OCR_AUTO_INTERVAL=6.0          # was 4.0 → fewer OCR passes
      - OCR_MAX_BASE_WIDTH=640
      - OCR_SCALE_FACTOR=1.0
      - OCR_MAX_RESULTS=6              # was 10 → less post-processing
      - OCR_VARIANT_THRESHOLD=off      # keep off for now (see Patch 2)

      # Reduce work
      - OCR_LANGS=en                   # was en,ru; switch to en for most games
      - OCR_GAMMA=1.0                  # was 1.2; remove extra transform
      - OCR_FORCE_CPU=1

      # Kill debug I/O
      - OCR_DEBUG_SAVE=0               # was 1
      - OCR_DEBUG_DIR=/tmp/ocr_debug   # keep, but it won’t spam

      # Critical: cap thread fan-out (Paddle/OpenMP/BLAS)
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - VECLIB_MAXIMUM_THREADS=1
      - NUMEXPR_NUM_THREADS=1
```

Apply:
```bash
cd ~/self-gaming
sudo docker compose up -d --force-recreate ocr_easy_agent
sudo docker stats --no-stream | head -n 12
```

Expected: `ocr_easy_agent` should drop from **~150–320% CPU** to **tens of %** average (spiky but much lower).

---

## Patch 2 — Code: “don’t OCR identical frames” gate (universal & powerful)

### Edit: `/app/ocr_easy_agent.py`
Implement a cheap frame-change detector before running PaddleOCR.
Simplest robust approach: compute a tiny grayscale hash and skip OCR if unchanged.

Pseudo-implementation (Codex should place it near the auto-loop):

```python
# new imports
import numpy as np
import cv2
import hashlib
import time

LAST_HASH = None
LAST_OCR_TS = 0.0

def quick_frame_hash(bgr, w=160):
    h = int(bgr.shape[0] * (w / bgr.shape[1]))
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # blur reduces noise-induced hash changes
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return hashlib.md5(gray.tobytes()).hexdigest()

def should_run_ocr(bgr, min_interval, force_every=30.0):
    global LAST_HASH, LAST_OCR_TS
    now = time.time()

    # Always allow a periodic OCR to avoid “stuck” states
    if now - LAST_OCR_TS > force_every:
        LAST_OCR_TS = now
        LAST_HASH = None
        return True

    h = quick_frame_hash(bgr)
    if LAST_HASH == h:
        return False

    LAST_HASH = h
    # also respect min interval
    if now - LAST_OCR_TS < min_interval:
        return False

    LAST_OCR_TS = now
    return True
```

Then in the loop:
```python
frame = read_latest_frame()  # BGR
if not should_run_ocr(frame, min_interval=float(os.getenv("OCR_AUTO_INTERVAL","6")), force_every=30.0):
    publish_heartbeat(empty=True)  # optional
    continue

# run PaddleOCR here
```
This is the single biggest universal CPU saver because most game frames don’t change text every 4 seconds.

---

## Patch 3 — Make timeouts impossible: publish heartbeat even when empty
Ensure `ocr_easy_agent` (or `ocr_agent` proxy) publishes at least every N seconds:
- `{ "text": "", "empty": true, "ts": ... }` when there’s no text

This keeps downstream subscribers from timing out and makes the pipeline deterministic.

---

## Verification checklist
Run these after applying Patch 1 (and after Patch 2 if you implement it):

```bash
# env check
sudo docker exec ocr_easy_agent env | egrep -i 'OCR_|OMP_|OPENBLAS|MKL|NUMEXPR'

# cpu check
sudo docker stats --no-stream | head -n 12

# message cadence check (10 sec window)
timeout 10s mosquitto_sub -h localhost -t ocr_easy/text -v
timeout 10s mosquitto_sub -h localhost -t ocr/text -v
```

Target steady-state:
- `ocr_easy_agent` average CPU < ~50% (spikes allowed)
- `simple_ocr_agent` < ~80% (you already reduced it nicely)
- swap usage should slowly decline or stop growing

---

## If you need RU sometimes (hybrid language mode)
Keep radar in `en`, and only switch to `en,ru` when onboarding detects Cyrillic-heavy UI.
(That can be a command on `ocr_easy/cmd` topic if your agent supports it.)

---

## 3) Policy Brain + event gating task
## Codex Task: Self_Gaming — ресурс-оптимизация + policy_brain (мягкая интеграция)

### Контекст / цель
Проект уже зрелый (MQTT-агенты, teacher/policy/act pipeline). Нужно:
1) Сильно снизить CPU/GPU/память без потери функционала за счёт **событийности** (не “всё на каждый кадр”), устранения лишних перекодировок, и ограничения потоков OCR.
2) Добавить новый агент **agents/policy_brain.py** как “мозг” (пока заглушка), который слушает `scene/state`, кодирует JSON в фиксированный вектор, прогоняет через dummy-модель, публикует в `policy_brain/cmd`, и сравнивает с реальным `act/cmd` (если оно есть), логируя ошибку.

---

# A) Оптимизация ресурсов (80/20)

## A1. Событийная перцепция: “не обрабатывать одинаковое”
### Требование
Добавить дешёвый “fingerprint” сцены и гейты, чтобы OCR/YOLO/teacher не дергались на практически одинаковых кадрах.

### Реализация
1) Ввести модуль `agents/utils/scene_fingerprint.py`:
   - функция `compute_fingerprint(frame: np.ndarray) -> str`:
     - downscale до ~64x64, grayscale, blur, затем `md5` от байтов
   - функция `should_process(prev_fp, new_fp, ttl_sec, force=False)`.

2) OCR agent:
   - добавить env:
     - `OCR_GATE_ENABLE=1`
     - `OCR_GATE_TTL_SEC=0.5` (или 1.0)
   - если fingerprint не изменился и TTL не истек — **не запускать OCR**, публиковать прошлый результат (или вообще пропускать публикацию, если потребители умеют ждать).

3) YOLO / object_detection_agent:
   - env:
     - `DETECT_GATE_ENABLE=1`
     - `DETECT_GATE_TTL_SEC=0.2..0.5`
   - аналогично: не детектить на одинаковом кадре; лучше пропустить/использовать прошлый результат.

4) teacher_agent:
   - добавить “тормоз” по частоте:
     - `TEACHER_MAX_HZ=0.5` (раз в 2 сек) + `TEACHER_BURST_ON_STUCK=1`
   - “stuck” детектор: N повторов одинакового act/cmd или reward не меняется T секунд → разрешить внеочередной запрос к teacher.

### Acceptance criteria
- При неподвижной сцене OCR/YOLO/teacher почти не работают (CPU падает заметно).
- Логи показывают “skipped due to fingerprint gate”.

---

## A2. Ограничение потоков OCR (главный CPU-хищник)
### Требование
Стабильно ограничить параллелизм (OpenMP/BLAS), чтобы OCR не забивал CPU.

### Реализация
В entrypoint OCR-контейнера или прямо в коде агента (до импорта тяжёлых libs) выставить:
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

Плюс:
- `OCR_DEBUG_SAVE=0` по умолчанию (не сохранять картинки/артефакты).
- Языки: `OCR_LANG=en` по умолчанию (если ru не нужен постоянно — включать явно).

### Acceptance criteria
- OCR больше не уходит в 150%+ CPU на пустом/статичном экране.
- Конфигурация видна в логах при старте.

---

## A3. Убрать лишние перекодировки видео
### Требование
Не гонять основную перцепцию через HTTP MJPEG feed. Источник должен быть напрямую `/dev/video0` (V4L2) или shared memory pipeline.

### Реализация (минимальная, без ломания)
1) В `docker-compose.yml`:
   - пробросить `/dev/video0` в контейнер DeepStream/perception где нужно.
   - добавить env `DS_SOURCE=v4l2` и `DS_V4L2_DEVICE=/dev/video0`
2) Внутри perception/deepstream:
   - если `DS_SOURCE=v4l2` → собирать gstreamer pipeline с v4l2src (MJPG/YUYV autodetect) и **nvvidconv/nvvideoconvert**.
   - HTTP viewer оставить как optional debug, но не как dependency.

### Acceptance criteria
- При отключенном HTTP video_feed перцепция продолжает работать (если камера доступна).
- FPS стабильнее, CPU меньше.

---

# B) Новый “Мозг” без ломания — agents/policy_brain.py

## B1. Что делает policy_brain
1) Подписывается на `scene/state` (JSON).
2) Кодирует JSON → фиксированный вектор `torch.Tensor` (или numpy, но лучше torch если уже есть).
3) Прогоняет через dummy model (заглушка).
4) Публикует результат в `policy_brain/cmd` (отдельно от `act/cmd`).
5) Подписывается на `act/cmd` (если появляется) и сравнивает с output policy_brain → логирует ошибку/метрики.

## B2. Требования к векторизации (самая важная часть)
Сделать **стабильную фиксированную размерность** и **детерминированность**, чтобы потом можно было заменить dummy на настоящую модель без изменения интерфейсов.

### Предложение формата вектора (пример)
`VECTOR_SIZE=512` (настраиваемо env).

Схема кодирования:
- Числовые признаки (скаляры):
  - brightness, reward_last, cursor_x, cursor_y, fps, latency_ms и т.п. (если есть)
  - нормализовать в [0,1] или z-score с clamp
- Категориальные/строковые:
  - `scene_name`, `ui_mode`, `game_state` и любые “лейблы” → feature hashing (например, murmur/md5) в несколько позиций вектора
- OCR текст:
  - взять top-K токенов (split по пробелам/неалфавитным), привести к lower, обрезать
  - feature hashing в “текстовый” сегмент вектора
- Объекты детектора:
  - для каждого объекта: (class_id hashed, conf, bbox normalized)
  - агрегировать: top-N по conf, остальное суммировать в “классовые корзины”

Важно:
- Порядок полей не должен ломать вектор.
- Если поле отсутствует — заполнять нулями.
- Версию схемы добавить в лог: `VECTOR_SCHEMA_VERSION=1`.

## B3. Dummy model
Сделать класс `DummyPolicyModel`:
- вход: `tensor [VECTOR_SIZE]` или `[1, VECTOR_SIZE]`
- выход: структура команды, например:
  ```json
  {"ts":..., "source":"policy_brain", "action":{"type":"noop"}, "confidence":0.0}
  ```
или (лучше) “универсальная” команда:
- `action_type`: one of `noop|mouse_move|mouse_click|key_press|combo`
- параметры: dx/dy или x/y, button, key, duration_ms

Пока dummy:
- `noop` всегда
- или простая эвристика: если в OCR есть “continue|ok|next” → click_center (опционально; можно оставить чистый noop)

## B4. Сравнение с act/cmd и лог ошибок
Подписаться на `act/cmd`:
- На каждом `act/cmd` сравнить с последним `policy_brain/cmd`, ближайшим по времени (окно ±1 сек).
- Метрика:
  - `match_action_type` (0/1)
  - если move/click: расстояние по координатам (L2)
  - если key: совпадение key
- Писать в лог:
  - `brain_vs_act: ts_brain=..., ts_act=..., match=..., dist=..., act=..., brain=...`
- Опционально публиковать метрики в `policy_brain/metrics`.

## B5. Конфиги env
Добавить (с дефолтами):
- `POLICY_BRAIN_ENABLE=1`
- `POLICY_BRAIN_VECTOR_SIZE=512`
- `POLICY_BRAIN_SCHEMA_VERSION=1`
- `POLICY_BRAIN_PUB_TOPIC=policy_brain/cmd`
- `POLICY_BRAIN_SCENE_TOPIC=scene/state`
- `POLICY_BRAIN_ACT_TOPIC=act/cmd`
- `POLICY_BRAIN_LOG_LEVEL=INFO`
- `POLICY_BRAIN_DEVICE=cpu` (пока cpu)

## B6. Интеграция в compose
- Добавить сервис `policy_brain_agent` (или как у вас принято), подключить к MQTT, передать env.
- Не менять существующую игру: `act/cmd` по-прежнему делает текущий policy/teacher.

## B7. Тесты / проверки
1) Юнит-тест на векторизацию:
- Один и тот же JSON → одинаковый tensor
- Разные JSON → разный tensor
- Размер ровно `VECTOR_SIZE`

2) Интеграционный smoke:
- Запустить MQTT + policy_brain, вручную publish sample `scene/state`
- Проверить что публикуется `policy_brain/cmd`

3) Сравнение:
- Когда игра активна и публикует `act/cmd`, логи сравнения появляются.

---

# C) Definition of Done (DoD)
- PR/patch с изменениями:
  - `agents/policy_brain.py`
  - `agents/utils/state_vectorizer.py` (или подобное)
  - `agents/utils/scene_fingerprint.py`
  - изменения OCR/YOLO/teacher для gating и throttling (минимально инвазивно)
  - docker-compose: новый сервис policy_brain + env + (если делаете) V4L2/DeepStream источник
- Документация:
  - README секция “Resource Optimization”
  - README секция “Policy Brain (shadow mode)”
- Логи при старте печатают ключевые env и версии схем.

---

## Примечание для Codex
Если schema `scene/state` плавает — векторизатор должен быть робастным: рекурсивно “плющить” JSON в пары `path=value`, и через feature hashing складывать в фиксированный вектор. Тогда интеграция заработает сразу, а позже можно уточнить schema без изменения интерфейсов.

---

## 4) Definition of Done (combined)
- DeepStream (`perception_ds`) uses `DS_SOURCE=v4l2` with `/dev/videoX`; does **not** crash/restart when HTTP viewer is down.
- OCR CPU reduced via thread caps + debug off + optional `en` default; frame-change gate skips OCR on identical frames; heartbeat messages prevent timeouts.
- Perception stack is more event-driven (gates for OCR/YOLO/teacher).
- New agent `agents/policy_brain.py` added in shadow-mode:
  - subscribes `scene/state`
  - vectorizes deterministically to fixed size tensor
  - runs dummy model
  - publishes to `policy_brain/cmd`
  - subscribes `act/cmd` and logs error metrics comparing outputs
- Docs updated: brief README section for V4L2, OCR mode, policy_brain shadow-mode; plus any generated overview if implemented.

