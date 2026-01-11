#!/usr/bin/env python3
import inspect
import json
import logging
import math
import os
import random
import signal
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

# --- Constants ---
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
NUMERIC_DIM, OBJECT_HIST_DIM, TEXT_EMBED_DIM = 32, 32, 64
FRAME_HEIGHT, FRAME_WIDTH = int(os.getenv("TRAIN_FRAME_HEIGHT", "96")), int(os.getenv("TRAIN_FRAME_WIDTH", "54"))
FRAME_SHAPE = (3, FRAME_HEIGHT, FRAME_WIDTH)
TEACHER_KL_WEIGHT = float(os.getenv("TEACHER_KL_WEIGHT", "1.0"))
REWARD_WEIGHT = float(os.getenv("REWARD_WEIGHT", "0.1"))
LEARNING_RATE = float(os.getenv("TRAIN_LR", "0.003"))
VALUE_LOSS_WEIGHT = float(os.getenv("VALUE_LOSS_WEIGHT", "0.1"))
BACKBONE_PATH = Path(os.getenv("BACKBONE_WEIGHTS_PATH", "/mnt/ssd/models/backbone/backbone.pt"))
POLICY_HEAD_PATH = Path(os.getenv("POLICY_HEAD_WEIGHTS_PATH", "/mnt/ssd/models/heads/ppo/policy_head.pt"))
VALUE_HEAD_PATH = Path(os.getenv("VALUE_HEAD_WEIGHTS_PATH", "/mnt/ssd/models/heads/ppo/value_head.pt"))
LABEL_MAP_PATH = Path(os.getenv("POLICY_LABEL_MAP_PATH", "/mnt/ssd/models/heads/ppo/label_map.json"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
logging.basicConfig(level=os.getenv("TRAIN_LOG_LEVEL", "INFO").upper(), format="[train_manager] %(levelname)s %(message)s")
logger = logging.getLogger("train_manager")
stop_event = threading.Event()

for p in [MODEL_DIR, RECORDER_DIR, BACKBONE_PATH.parent, POLICY_HEAD_PATH.parent, VALUE_HEAD_PATH.parent, LABEL_MAP_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class EpisodeDataset(Dataset):
    def __init__(self, tensor_dict):
        self.features = torch.from_numpy(tensor_dict["features"])
        self.labels = torch.from_numpy(tensor_dict["labels"])
        self.teacher = torch.from_numpy(tensor_dict["teacher"])
        self.rewards = torch.from_numpy(tensor_dict["rewards"])

    def __len__(self): return self.features.size(0)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.teacher[idx], self.rewards[idx]

class TrainManagerAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="train_manager", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _publish_status(self, payload):
        logger.debug("publish_status: %s", payload)
        self.client.publish(TRAIN_STATUS_TOPIC, json.dumps(payload))

    def _publish_checkpoint(self, payload):
        logger.debug("publish_checkpoint: %s", payload)
        self.client.publish(CHECKPOINT_TOPIC, json.dumps(payload))

    def _on_connect(self, client, userdata, flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(TRAIN_JOB_TOPIC, 0)])
            self._publish_status({"ok": True, "event": "train_manager_ready"})
            logger.info("Connected to MQTT broker; subscribed to %s", TRAIN_JOB_TOPIC)
        else:
            self._publish_status({"ok": False, "event": "connect_failed", "code": _as_int(rc)})
            logger.error("Failed to connect to MQTT (%s)", rc)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            self._publish_status({"ok": False, "event": "job_failed", "error": "invalid_json"})
            logger.error("Invalid JSON on topic %s", msg.topic)
            return
        if msg.topic == TRAIN_JOB_TOPIC and data.get("ok"):
            if data.get("mode") == "world_model_experiment" or data.get("target") == "world_model":
                logger.info("Skipping world-model job %s for dedicated agent", data.get("job_id"))
                return
            logger.info("Received job payload: %s", {k: data.get(k) for k in ('job_id', 'dataset', 'mode')})
            self._handle_job(data)

    def _handle_job(self, job):
        job_id = job.get("job_id") or f"job_{int(time.time())}"
        logger.info("Queueing job %s", job_id)
        thread = threading.Thread(target=self._train_model, args=(job_id, job), daemon=True)
        thread.start()

    def _train_model(self, job_id, job):
        logger.info("Starting train_model for job %s", job_id)
        try:
            dataset_id = job.get("dataset")
            files = self._load_samples(dataset_id)
            dataset, label_map = self._prepare_dataset(files)
            if dataset is None:
                raise RuntimeError("Insufficient data after preparation")

            num_samples, num_classes = dataset["features"].shape[0], len(label_map)
            teacher_cfg, reward_weight = job.get("teacher", {}), float(job.get("reward_weight", REWARD_WEIGHT))
            alpha_start, decay_steps = float(teacher_cfg.get("alpha_start", 0.0)), int(teacher_cfg.get("decay_steps", max(1, num_samples * EPOCHS)))
            teacher_min_alpha, teacher_weight = float(teacher_cfg.get("alpha_min", 0.0)), float(teacher_cfg.get("kl_weight", TEACHER_KL_WEIGHT))
            
            def current_alpha(step: int) -> float:
                if alpha_start <= 0: return 0.0
                return max(teacher_min_alpha, alpha_start * (1.0 - min(step / float(decay_steps), 1.0)))

            use_rnn = job.get("use_rnn", False)
            backbone = self._build_backbone(use_rnn).to(DEVICE)
            policy_head, value_head = nn.Linear(backbone.output_dim, num_classes).to(DEVICE), nn.Linear(backbone.output_dim, 1).to(DEVICE)
            optimizer = torch.optim.Adam(list(backbone.parameters()) + list(policy_head.parameters()) + list(value_head.parameters()), lr=LEARNING_RATE)

            dataloader = DataLoader(EpisodeDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)
            mode = job.get("mode", "ppo_baseline")
            self._publish_status({"ok": True, "event": "job_started", "job_id": job_id, "samples": num_samples, "mode": mode})
            logger.info("Job %s started: samples=%s labels=%s mode=%s", job_id, num_samples, num_classes, mode)
            
            global_step = 0
            for epoch in range(1, EPOCHS + 1):
                epoch_loss, correct = 0.0, 0
                for features, labels, teacher, reward in dataloader:
                    features, labels, teacher, reward = features.to(DEVICE).float(), labels.to(DEVICE).long(), teacher.to(DEVICE).long(), reward.to(DEVICE).float()
                    optimizer.zero_grad()
                    final_state = backbone(torch.zeros(features.size(0), *FRAME_SHAPE, device=DEVICE), features)
                    logits, values = policy_head(final_state), value_head(final_state).squeeze(-1)
                    
                    ce_loss = F.cross_entropy(logits, labels, reduction="none")
                    sample_weights = torch.ones_like(ce_loss) + (-reward_weight) * reward.clamp(-1.0, 1.0) if reward_weight != 0 else torch.ones_like(ce_loss)
                    loss = (ce_loss * sample_weights).mean() + VALUE_LOSS_WEIGHT * F.mse_loss(values, reward)
                    
                    teacher_alpha = current_alpha(global_step)
                    if teacher_weight > 0 and teacher_alpha > 0:
                        mask = teacher >= 0
                        if torch.any(mask):
                            teacher_targets = F.one_hot(teacher[mask], num_classes).float()
                            kl_loss = F.kl_div(F.log_softmax(logits[mask], dim=-1), teacher_targets, reduction="batchmean")
                            loss += teacher_weight * teacher_alpha * kl_loss
                    
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
                        epoch_loss += float(loss.item()) * features.size(0)
                    global_step += 1
                
                avg_loss, acc = epoch_loss / max(num_samples, 1), correct / max(num_samples, 1)
                self._publish_status({"ok": True, "event": "job_progress", "job_id": job_id, "epoch": epoch, "epochs": EPOCHS,
                                     "loss": round(avg_loss, 4), "accuracy": round(acc, 4), "teacher_alpha": round(current_alpha(global_step), 4)})
                logger.info("Job %s epoch %s/%s loss=%.4f acc=%.4f", job_id, epoch, EPOCHS, avg_loss, acc)

            torch.save(backbone.state_dict(), BACKBONE_PATH)
            torch.save(policy_head.state_dict(), POLICY_HEAD_PATH)
            torch.save(value_head.state_dict(), VALUE_HEAD_PATH)
            LABEL_MAP_PATH.write_text(json.dumps(label_map))
            self._publish_status({"ok": True, "event": "job_finished", "job_id": job_id, "model": str(POLICY_HEAD_PATH)})
            self._publish_checkpoint({"job_id": job_id, "backbone_path": str(BACKBONE_PATH), "policy_head_path": str(POLICY_HEAD_PATH),
                                      "value_head_path": str(VALUE_HEAD_PATH), "label_map_path": str(LABEL_MAP_PATH), "timestamp": time.time()})
            logger.info("Job %s finished; checkpoints saved to %s", job_id, POLICY_HEAD_PATH.parent)

        except Exception as e:
            logger.error("Training job %s failed: %s", job_id, e, exc_info=True)
            self._publish_status({"ok": False, "event": "job_failed", "job_id": job_id, "error": str(e)})

    def _build_backbone(self, use_rnn: bool) -> Backbone:
        try:
            sig = inspect.signature(Backbone)
            if "use_rnn" in sig.parameters:
                return Backbone(frame_shape=FRAME_SHAPE, non_visual_dim=NON_VISUAL_DIM, use_rnn=use_rnn)
        except (TypeError, ValueError):
            pass
        if use_rnn:
            logger.warning("Backbone does not support use_rnn; ignoring.")
        return Backbone(frame_shape=FRAME_SHAPE, non_visual_dim=NON_VISUAL_DIM)

    def _load_samples(self, dataset_id=None, max_samples=2000):
        path_glob = (RECORDER_DIR / dataset_id).glob("*.json") if dataset_id and (RECORDER_DIR / dataset_id).is_dir() else RECORDER_DIR.glob("*.json")
        files = sorted(path_glob)
        if not files: logger.warning("No samples found in %s", dataset_id or RECORDER_DIR)
        random.shuffle(files)
        limited = files[:max_samples]
        logger.info("Loaded %s samples (limited to %s)", len(limited), max_samples)
        return limited

    def _prepare_dataset(self, files):
        features, labels, teacher_labels, rewards, label_map = [], [], [], [], {}
        for path in files:
            try:
                data = json.loads(path.read_text())
                scene, action_payload = data.get("scene", {}), data.get("action")
                action = str(action_payload.get("action") or action_payload) if isinstance(action_payload, dict) else str(action_payload)
                if not action: continue
                
                features.append(self._encode_scene(scene))
                labels.append(label_map.setdefault(action, len(label_map)))
                
                teacher_meta = data.get("teacher", {})
                teacher_action = teacher_meta.get("action") or teacher_meta.get("text")
                teacher_labels.append(label_map.setdefault(str(teacher_action), len(label_map)) if teacher_action else -1)
                
                reward_value = data.get("reward", {}).get("value", 0.0)
                rewards.append(float(reward_value))
            except Exception: continue

        if not features or len(label_map) < 2:
            logger.warning("prepare_dataset produced no usable samples (features=%s, labels=%s)", len(features), len(label_map))
            return None, None
            
        dataset = {"features": np.asarray(features, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int64),
                   "teacher": np.asarray(teacher_labels, dtype=np.int64), "rewards": np.asarray(rewards, dtype=np.float32)}
        logger.info("Prepared dataset with %s samples, %s distinct labels", len(features), len(label_map))
        return dataset, label_map

    def _encode_scene(self, scene):
        vector = np.zeros(NON_VISUAL_DIM, dtype=np.float32)
        numeric = np.zeros(NUMERIC_DIM, dtype=np.float32)
        
        trend = scene.get("trend") or []
        try: trend_vals = [float(x) for x in trend if isinstance(x, (int, float))]
        except Exception: trend_vals = []
        
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
        vector[:NUMERIC_DIM] = numeric

        hist = np.zeros(OBJECT_HIST_DIM, dtype=np.float32)
        # OBJECT_CLASS_BUCKETS logic inline to avoid external dependency if needed, 
        # or use a consistent mapping. Here we use a simple hash or predefined buckets.
        # Re-using the logic from previous version for consistency:
        buckets = {
            "enemy_melee": 0, "enemy_ranged": 1, "boss": 2, "projectile": 3,
            "loot_currency": 4, "loot_rare": 5, "portal": 6, "npc": 7, "chest": 8, "hazard": 9
        }
        for obj in objects:
            label = str(obj.get("class") or obj.get("label") or "unknown").lower()
            idx = buckets.get(label, hash(label) % OBJECT_HIST_DIM)
            hist[idx] += 1.0
        
        start = NUMERIC_DIM
        vector[start : start + OBJECT_HIST_DIM] = hist

        text_bins = np.zeros(TEXT_EMBED_DIM, dtype=np.float32)
        for entry in text_entries:
            for token in str(entry).lower().split():
                idx = hash(token) % TEXT_EMBED_DIM
                text_bins[idx] += 1.0
        vector[start + OBJECT_HIST_DIM :] = text_bins
        return vector

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Train manager shut down.")

def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down.", signum)
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = TrainManagerAgent()
    agent.run()

if __name__ == "__main__":
    main()
