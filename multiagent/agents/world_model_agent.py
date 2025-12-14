#!/usr/bin/env python3
"""World-model agent for experimental Dreamer-style runs (baseline autoencoder)."""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=os.getenv("WORLD_MODEL_LOG_LEVEL", "INFO"))
logger = logging.getLogger("world_model_agent")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TRAIN_JOB_TOPIC = os.getenv("TRAIN_JOB_TOPIC", "train/jobs")
TRAIN_STATUS_TOPIC = os.getenv("TRAIN_STATUS_TOPIC", "train/status")
CHECKPOINT_TOPIC = os.getenv("CHECKPOINT_TOPIC", "train/checkpoints")
RECORDER_DIR = Path(os.getenv("RECORDER_DIR", "/mnt/ssd/datasets/episodes"))
MODEL_PATH = Path(os.getenv("WORLD_MODEL_PATH", "/mnt/ssd/models/heads/world_model/world_model.pt"))
NON_VISUAL_DIM = 128
BATCH_SIZE = int(os.getenv("WORLD_MODEL_BATCH", "64"))
EPOCHS = int(os.getenv("WORLD_MODEL_EPOCHS", "5"))
LR = float(os.getenv("WORLD_MODEL_LR", "0.001"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_ERROR_TOPIC = os.getenv("PRED_ERROR_TOPIC", "world_model/pred_error")
PRED_ERROR_INTERVAL = float(os.getenv("PRED_ERROR_INTERVAL", "10"))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RECORDER_DIR.mkdir(parents=True, exist_ok=True)


def encode_scene(scene: Dict) -> np.ndarray:
    vec = np.zeros(NON_VISUAL_DIM, dtype=np.float32)
    text = scene.get("text") or []
    objects = scene.get("objects") or []
    vec[0] = float(scene.get("mean", 0.0))
    vec[1] = float(len(objects))
    vec[2] = float(len(text))
    idx = 3
    for entry in text:
        for token in str(entry).split():
            bucket = idx + (hash(token) % (NON_VISUAL_DIM - idx))
            vec[bucket] += 1.0
    return vec


def load_files(dataset_id: Optional[str]) -> list:
    if dataset_id:
        path = RECORDER_DIR / dataset_id
        if path.exists() and path.is_dir():
            return sorted(path.glob("*.json"))
    return sorted(RECORDER_DIR.glob("*.json"))


def prepare_dataset(files) -> Optional[np.ndarray]:
    features = []
    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        scene = data.get("scene") or {}
        features.append(encode_scene(scene))
    if not features:
        return None
    return np.asarray(features, dtype=np.float32)


class FeatureDataset(Dataset):
    def __init__(self, feats: np.ndarray):
        self.features = torch.from_numpy(feats)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        x = self.features[idx].float()
        return x, x


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256),
            nn.ReLU(True),
            nn.Linear(256, dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


class WorldModelAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="world_model", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self._stop = threading.Event()

    def on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(TRAIN_JOB_TOPIC, 0)])
            logger.info("world_model_agent ready")
        else:
            logger.error("world_model_agent failed to connect rc=%s", rc)

    def on_message(self, client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            return
        if not data.get("ok"):
            return
        mode = data.get("mode")
        target = data.get("target")
        if mode != "world_model_experiment" and target != "world_model":
            return
        threading.Thread(target=self.handle_job, args=(data,), daemon=True).start()

    def handle_job(self, job):
        job_id = job.get("job_id") or f"wm_{int(time.time())}"
        publish = lambda evt, **extra: self.client.publish(
            TRAIN_STATUS_TOPIC,
            json.dumps({"ok": True, "event": evt, "job_id": job_id, **extra}),
        )
        publish("world_model_started")
        files = load_files(job.get("dataset"))
        feats = prepare_dataset(files)
        if feats is None:
            publish("world_model_failed", error="no_data")
            return
        dataset = FeatureDataset(feats)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = AutoEncoder(NON_VISUAL_DIM).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        for epoch in range(1, EPOCHS + 1):
            total = 0.0
            for batch, target in loader:
                batch = batch.to(DEVICE)
                target = target.to(DEVICE)
                optim.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, target)
                loss.backward()
                optim.step()
                total += float(loss.item()) * batch.size(0)
            avg = total / max(len(dataset), 1)
            publish("world_model_progress", epoch=epoch, epochs=EPOCHS, loss=round(avg, 4))
        torch.save(model.state_dict(), MODEL_PATH)
        publish("world_model_finished", path=str(MODEL_PATH))
        self.client.publish(
            CHECKPOINT_TOPIC,
            json.dumps(
                {
                    "job_id": job_id,
                    "world_model_path": str(MODEL_PATH),
                    "timestamp": time.time(),
                }
            ),
        )

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        threading.Thread(target=self._publish_error_loop, daemon=True).start()
        try:
            self.client.loop_forever()
        finally:
            self._stop.set()

    def _publish_error_loop(self):
        """Publish placeholder prediction_error until real world model metrics are wired."""

        while not self._stop.is_set():
            time.sleep(max(1.0, PRED_ERROR_INTERVAL))
            # TODO: replace random error with actual embedding prediction metrics.
            error = random.uniform(0.0, 1.0)
            payload = {"ok": True, "error": round(error, 4), "timestamp": time.time()}
            try:
                self.client.publish(PRED_ERROR_TOPIC, json.dumps(payload))
            except Exception:
                logger.debug("failed to publish pred_error", exc_info=False)


def main():
    agent = WorldModelAgent()
    agent.run()


if __name__ == "__main__":
    main()
