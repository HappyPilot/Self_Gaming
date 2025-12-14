#!/usr/bin/env python3
import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.backbone import Backbone

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TRAIN_JOB_TOPIC = os.getenv("TRAIN_JOB_TOPIC", "train/jobs")
TRAIN_STATUS_TOPIC = os.getenv("TRAIN_STATUS_TOPIC", "train/status")
CHECKPOINT_TOPIC = os.getenv("CHECKPOINT_TOPIC", "train/checkpoints")
RECORDER_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/mnt/ssd/models"))
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
NON_VISUAL_DIM = 128
NUMERIC_DIM = 32
OBJECT_HIST_DIM = 32
TEXT_EMBED_DIM = 64
FRAME_HEIGHT = int(os.getenv("TRAIN_FRAME_HEIGHT", "96"))
FRAME_WIDTH = int(os.getenv("TRAIN_FRAME_WIDTH", "54"))
FRAME_SHAPE = (3, FRAME_HEIGHT, FRAME_WIDTH)
TEACHER_KL_WEIGHT = float(os.getenv("TEACHER_KL_WEIGHT", "1.0"))
REWARD_WEIGHT = float(os.getenv("REWARD_WEIGHT", "0.1"))
LOG_LEVEL = os.getenv("TRAIN_LOG_LEVEL", "INFO").upper()
LEARNING_RATE = float(os.getenv("TRAIN_LR", "0.003"))
BACKBONE_PATH = Path(
    os.getenv("BACKBONE_WEIGHTS_PATH", "/mnt/ssd/models/backbone/backbone.pt")
)
POLICY_HEAD_PATH = Path(
    os.getenv("POLICY_HEAD_WEIGHTS_PATH", "/mnt/ssd/models/heads/ppo/policy_head.pt")
)
VALUE_HEAD_PATH = Path(
    os.getenv("VALUE_HEAD_WEIGHTS_PATH", "/mnt/ssd/models/heads/ppo/value_head.pt")
)
LABEL_MAP_PATH = Path(
    os.getenv("POLICY_LABEL_MAP_PATH", "/mnt/ssd/models/heads/ppo/label_map.json")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALUE_LOSS_WEIGHT = float(os.getenv("VALUE_LOSS_WEIGHT", "0.1"))

logging.basicConfig(level=LOG_LEVEL, format="[train_manager] %(levelname)s %(message)s")
logger = logging.getLogger("train_manager")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RECORDER_DIR.mkdir(parents=True, exist_ok=True)
BACKBONE_PATH.parent.mkdir(parents=True, exist_ok=True)
POLICY_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
VALUE_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

client = mqtt.Client(client_id="train_manager", protocol=mqtt.MQTTv311)


def publish_status(payload):
    logger.debug("publish_status: %s", payload)
    client.publish(TRAIN_STATUS_TOPIC, json.dumps(payload))


def publish_checkpoint(payload):
    logger.debug("publish_checkpoint: %s", payload)
    client.publish(CHECKPOINT_TOPIC, json.dumps(payload))


def load_samples(dataset_id=None, max_samples=2000):
    if dataset_id:
        dataset_path = RECORDER_DIR / dataset_id
        if dataset_path.exists() and dataset_path.is_dir():
            files = sorted(dataset_path.glob("*.json"))
        else:
            files = sorted(RECORDER_DIR.glob("*.json"))
    else:
        files = sorted(RECORDER_DIR.glob("*.json"))
    if not files:
        logger.warning("No samples found in %s", dataset_id or RECORDER_DIR)
        return []
    random.shuffle(files)
    limited = files[:max_samples]
    logger.info("Loaded %s samples (limited to %s)", len(limited), max_samples)
    return limited


OBJECT_CLASS_BUCKETS: Dict[str, int] = {
    "enemy_melee": 0,
    "enemy_ranged": 1,
    "boss": 2,
    "projectile": 3,
    "loot_currency": 4,
    "loot_rare": 5,
    "portal": 6,
    "npc": 7,
    "chest": 8,
    "hazard": 9,
}


def _object_bin(label: str) -> int:
    key = label.lower()
    if key in OBJECT_CLASS_BUCKETS:
        return OBJECT_CLASS_BUCKETS[key]
    return hash(key) % OBJECT_HIST_DIM


def encode_scene(scene):
    vector = np.zeros(NON_VISUAL_DIM, dtype=np.float32)

    numeric = np.zeros(NUMERIC_DIM, dtype=np.float32)
    trend = scene.get("trend") or []
    try:
        trend_vals = [float(x) for x in trend if isinstance(x, (int, float))]
    except Exception:
        trend_vals = []
    mean = float(scene.get("mean", trend_vals[-1] if trend_vals else 0.0))
    objects = scene.get("objects") or []
    text_entries = scene.get("text") or []

    numeric[0] = mean
    numeric[1] = float(len(objects))
    numeric[2] = float(sum(1 for obj in objects if "enemy" in str(obj.get("class", "")).lower()))
    numeric[3] = float(sum(1 for obj in objects if "loot" in str(obj.get("class", "")).lower()))
    numeric[4] = float(len(text_entries))
    stats = scene.get("stats") or {}
    numeric[5] = float(stats.get("hp_pct", 0.0))
    numeric[6] = float(stats.get("mana_pct", 0.0))
    numeric[7] = float(stats.get("xp_pct", 0.0))
    # remaining numeric slots stay zero for future signals
    vector[:NUMERIC_DIM] = numeric

    hist = np.zeros(OBJECT_HIST_DIM, dtype=np.float32)
    for obj in objects:
        label = str(obj.get("class") or obj.get("label") or "unknown")
        hist[_object_bin(label)] += 1.0
    start = NUMERIC_DIM
    vector[start : start + OBJECT_HIST_DIM] = hist

    text_bins = np.zeros(TEXT_EMBED_DIM, dtype=np.float32)
    for entry in text_entries:
        for token in str(entry).lower().split():
            idx = hash(token) % TEXT_EMBED_DIM
            text_bins[idx] += 1.0
    vector[start + OBJECT_HIST_DIM :] = text_bins
    return vector


def prepare_dataset(files):
    """Load samples into numpy arrays including teacher and reward annotations."""

    features = []
    labels = []
    teacher_labels = []
    rewards = []
    label_map = {}

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        scene = data.get("scene") or {}
        action_payload = data.get("action")
        if isinstance(action_payload, dict):
            action = str(action_payload.get("action") or action_payload)
        else:
            action = str(action_payload)
        if not action:
            continue

        feat = encode_scene(scene)
        target_idx = label_map.setdefault(action, len(label_map))

        teacher_meta = data.get("teacher") or {}
        teacher_action = teacher_meta.get("action") or teacher_meta.get("text")
        if teacher_action:
            teacher_idx = label_map.setdefault(str(teacher_action), len(label_map))
        else:
            teacher_idx = -1

        reward_meta = data.get("reward") or {}
        reward_value = reward_meta.get("value", 0.0)
        try:
            reward_float = float(reward_value)
        except (TypeError, ValueError):
            reward_float = 0.0

        features.append(feat)
        labels.append(target_idx)
        teacher_labels.append(teacher_idx)
        rewards.append(reward_float)

    if not features or len(label_map) < 1:
        logger.warning(
            "prepare_dataset produced no usable samples (features=%s, labels=%s)",
            len(features),
            len(label_map),
        )
        return None, None

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    teacher_array = np.asarray(teacher_labels, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float32)
    logger.info(
        "Prepared dataset with %s samples, %s distinct labels", len(features), len(label_map)
    )
    dataset = {
        "features": X,
        "labels": y,
        "teacher": teacher_array,
        "rewards": reward_array,
    }
    return dataset, label_map


class EpisodeDataset(Dataset):
    def __init__(self, tensor_dict):
        self.features = torch.from_numpy(tensor_dict["features"])
        self.labels = torch.from_numpy(tensor_dict["labels"])
        self.teacher = torch.from_numpy(tensor_dict["teacher"])
        self.rewards = torch.from_numpy(tensor_dict["rewards"])

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.teacher[idx],
            self.rewards[idx],
        )


def train_model(job_id, job):
    logger.info("Starting train_model for job %s", job_id)
    dataset_id = job.get("dataset")
    files = load_samples(dataset_id)
    dataset, label_map = prepare_dataset(files)
    if dataset is None:
        publish_status(
            {"ok": False, "event": "job_failed", "job_id": job_id, "error": "insufficient_data"}
        )
        logger.error("Job %s failed: insufficient data", job_id)
        return

    num_samples = dataset["features"].shape[0]
    num_classes = len(label_map)

    teacher_cfg = job.get("teacher") or {}
    alpha_start = float(teacher_cfg.get("alpha_start", 0.0))
    decay_steps = int(teacher_cfg.get("decay_steps", max(1, num_samples * EPOCHS)))
    teacher_min_alpha = float(teacher_cfg.get("alpha_min", 0.0))
    teacher_weight = float(teacher_cfg.get("kl_weight", TEACHER_KL_WEIGHT))
    reward_weight = float(job.get("reward_weight", REWARD_WEIGHT))

    def current_alpha(step: int) -> float:
        if alpha_start <= 0:
            return 0.0
        progress = min(step / float(decay_steps), 1.0)
        value = alpha_start * (1.0 - progress)
        return max(teacher_min_alpha, value)

    backbone = Backbone(frame_shape=FRAME_SHAPE, non_visual_dim=NON_VISUAL_DIM).to(DEVICE)
    policy_head = nn.Linear(backbone.output_dim, num_classes).to(DEVICE)
    value_head = nn.Linear(backbone.output_dim, 1).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(policy_head.parameters()) + list(value_head.parameters()),
        lr=LEARNING_RATE,
    )

    dataset_torch = EpisodeDataset(dataset)
    dataloader = DataLoader(dataset_torch, batch_size=BATCH_SIZE, shuffle=True)

    mode = job.get("mode", "ppo_baseline")
    publish_status({"ok": True, "event": "job_started", "job_id": job_id, "samples": num_samples, "mode": mode})
    logger.info("Job %s started: samples=%s labels=%s mode=%s", job_id, num_samples, num_classes, mode)

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        correct = 0
        for batch_features, batch_labels, batch_teacher, batch_reward in dataloader:
            batch_features = batch_features.to(DEVICE).float()
            batch_labels = batch_labels.to(DEVICE).long()
            batch_teacher = batch_teacher.to(DEVICE).long()
            batch_reward = batch_reward.to(DEVICE).float()

            optimizer.zero_grad()
            frame_placeholder = torch.zeros(
                batch_features.size(0), *FRAME_SHAPE, device=DEVICE
            )
            final_state = backbone(frame_placeholder, batch_features)
            logits = policy_head(final_state)
            values = value_head(final_state).squeeze(-1)

            ce_loss = F.cross_entropy(logits, batch_labels, reduction="none")
            sample_weights = torch.ones_like(ce_loss)
            if reward_weight != 0:
                sample_weights = sample_weights + (-reward_weight) * batch_reward.clamp(-1.0, 1.0)
            base_loss = (ce_loss * sample_weights).mean()

            value_loss = F.mse_loss(values, batch_reward)
            loss = base_loss + VALUE_LOSS_WEIGHT * value_loss

            teacher_alpha = current_alpha(global_step)
            if teacher_weight > 0 and teacher_alpha > 0:
                mask = batch_teacher >= 0
                if torch.any(mask):
                    teacher_targets = F.one_hot(batch_teacher[mask], num_classes).float()
                    student_log = F.log_softmax(logits[mask], dim=-1)
                    kl_loss = F.kl_div(student_log, teacher_targets, reduction="batchmean")
                    loss = loss + teacher_weight * teacher_alpha * kl_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == batch_labels).sum().item())
                epoch_loss += float(loss.item()) * batch_features.size(0)
            global_step += 1

        avg_loss = epoch_loss / max(num_samples, 1)
        acc = correct / max(num_samples, 1)
        publish_status(
            {
                "ok": True,
                "event": "job_progress",
                "job_id": job_id,
                "epoch": epoch,
                "epochs": EPOCHS,
                "loss": round(float(avg_loss), 4),
                "accuracy": round(float(acc), 4),
                "teacher_alpha": round(current_alpha(global_step), 4),
                "reward_weight": reward_weight,
                "mode": mode,
            }
        )
        logger.info(
            "Job %s epoch %s/%s loss=%.4f acc=%.4f teacher_alpha=%.4f",
            job_id,
            epoch,
            EPOCHS,
            avg_loss,
            acc,
            current_alpha(global_step),
        )

    torch.save(backbone.state_dict(), BACKBONE_PATH)
    torch.save(policy_head.state_dict(), POLICY_HEAD_PATH)
    torch.save(value_head.state_dict(), VALUE_HEAD_PATH)
    LABEL_MAP_PATH.write_text(json.dumps(label_map))
    publish_status(
        {
            "ok": True,
            "event": "job_finished",
            "job_id": job_id,
            "model": str(POLICY_HEAD_PATH),
            "teacher": {"alpha_start": alpha_start, "decay_steps": decay_steps},
            "reward_weight": reward_weight,
            "mode": mode,
        }
    )
    publish_checkpoint(
        {
            "job_id": job_id,
            "backbone_path": str(BACKBONE_PATH),
            "policy_head_path": str(POLICY_HEAD_PATH),
            "value_head_path": str(VALUE_HEAD_PATH),
            "label_map_path": str(LABEL_MAP_PATH),
            "timestamp": time.time(),
        }
    )
    logger.info(
        "Job %s finished; checkpoints saved to %s", job_id, POLICY_HEAD_PATH.parent
    )


def handle_job(job):
    job_id = job.get("job_id") or f"job_{int(time.time())}"
    logger.info("Queueing job %s", job_id)
    thread = threading.Thread(target=train_model, args=(job_id, job), daemon=True)
    thread.start()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe([(TRAIN_JOB_TOPIC, 0)])
        publish_status({"ok": True, "event": "train_manager_ready"})
        logger.info("Connected to MQTT broker; subscribed to %s", TRAIN_JOB_TOPIC)
    else:
        publish_status({"ok": False, "event": "connect_failed", "code": int(rc)})
        logger.error("Failed to connect to MQTT (%s)", rc)


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
    except Exception:
        publish_status({"ok": False, "event": "job_failed", "error": "invalid_json"})
        logger.error("Invalid JSON on topic %s: %s", msg.topic, payload[:120])
        return

    if msg.topic == TRAIN_JOB_TOPIC and data.get("ok"):
        if data.get("mode") == "world_model_experiment" or data.get("target") == "world_model":
            logger.info("Skipping world-model job %s for dedicated agent", data.get("job_id"))
            return
        logger.info("Received job payload on %s: %s", TRAIN_JOB_TOPIC, {k: data.get(k) for k in ('job_id', 'dataset', 'mode')})
        handle_job(data)


def main():
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 30)
    client.loop_forever()


if __name__ == "__main__":
    main()
