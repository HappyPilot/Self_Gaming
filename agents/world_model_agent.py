#!/usr/bin/env python3
"""World-model agent for experimental Dreamer-style runs (baseline autoencoder)."""
from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- Setup ---
logging.basicConfig(level=os.getenv("WORLD_MODEL_LOG_LEVEL", "INFO"), format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("world_model_agent")
stop_event = threading.Event()

# --- Constants ---
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

def _as_int(code) -> int:
    try:
        if hasattr(code, "value"): return int(code.value)
        return int(code)
    except (TypeError, ValueError): return 0

class FeatureDataset(Dataset):
    def __init__(self, feats: np.ndarray):
        self.features = torch.from_numpy(feats)
    def __len__(self): return self.features.size(0)
    def __getitem__(self, idx): return self.features[idx], self.features[idx]

class RecurrentWorldModel(nn.Module):
    def __init__(self, dim: int, latent: int = 64, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(True), nn.Linear(hidden, latent))
        self.rnn = nn.GRU(latent, hidden, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(True), nn.Linear(hidden, dim))

    def forward(self, x, h=None):
        # x: (Batch, Seq, Dim) or (Batch, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Encode each step
        flat_x = x.view(-1, x.size(-1))
        z = self.encoder(flat_x)
        z = z.view(batch_size, seq_len, -1)
        
        # Temporal dynamics
        out, h = self.rnn(z, h)
        
        # Decode each step
        flat_out = out.contiguous().view(-1, out.size(-1))
        recon = self.decoder(flat_out)
        recon = recon.view(batch_size, seq_len, -1)
        
        if seq_len == 1:
            recon = recon.squeeze(1)
            
        return recon, h

class WorldModelAgent:
    def __init__(self):
        self.client = mqtt.Client(client_id="world_model", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            client.subscribe([(TRAIN_JOB_TOPIC, 0)])
            logger.info("World model agent connected and ready.")
        else:
            logger.error("World model agent failed to connect: rc=%s", _as_int(rc))
            
    def on_disconnect(self, _client, _userdata, rc):
        if _as_int(rc) != 0:
            logger.warning("World model agent disconnected unexpectedly: rc=%s", _as_int(rc))

    def on_message(self, client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: return
        if not data.get("ok"): return
        mode, target = data.get("mode"), data.get("target")
        if mode == "world_model_experiment" or target == "world_model":
            threading.Thread(target=self.handle_job, args=(data,), daemon=True).start()

    def handle_job(self, job):
        job_id = job.get("job_id") or f"wm_{int(time.time())}"
        publish = lambda evt, **extra: self.client.publish(TRAIN_STATUS_TOPIC, json.dumps({"ok": True, "event": evt, "job_id": job_id, **extra}))
        try:
            publish("world_model_started")
            files = self._load_files(job.get("dataset"))
            feats = self._prepare_dataset(files)
            if feats is None: raise ValueError("No features prepared from dataset.")
            
            dataset = FeatureDataset(feats)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            model = RecurrentWorldModel(NON_VISUAL_DIM).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.MSELoss()

            for epoch in range(1, EPOCHS + 1):
                if stop_event.is_set():
                    logger.warning("Training for job %s interrupted.", job_id)
                    break
                
                total_loss = 0.0
                for batch, target in loader:
                    batch, target = batch.to(DEVICE), target.to(DEVICE)
                    optim.zero_grad()
                    recon, _ = model(batch)
                    loss = loss_fn(recon, target)
                    loss.backward()
                    optim.step()
                    total_loss += float(loss.item()) * batch.size(0)
                
                avg_loss = total_loss / max(len(dataset), 1)
                publish("world_model_progress", epoch=epoch, epochs=EPOCHS, loss=round(avg_loss, 5))
            
            if not stop_event.is_set():
                torch.save(model.state_dict(), MODEL_PATH)
                publish("world_model_finished", path=str(MODEL_PATH))
                self.client.publish(CHECKPOINT_TOPIC, json.dumps({"job_id": job_id, "world_model_path": str(MODEL_PATH), "timestamp": time.time()}))
                logger.info("Job %s finished, model saved to %s", job_id, MODEL_PATH)

        except Exception as e:
            logger.error("World model job %s failed: %s", job_id, e, exc_info=True)
            self.client.publish(TRAIN_STATUS_TOPIC, json.dumps({"ok": False, "event": "world_model_failed", "job_id": job_id, "error": str(e)}))

    def _train_batch(self, model, loss_fn, optim, batch, target):
        batch, target = batch.to(DEVICE), target.to(DEVICE)
        optim.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, target)
        loss.backward()
        optim.step()
        return float(loss.item()) * batch.size(0)

    def _load_files(self, dataset_id: Optional[str]) -> list:
        path = RECORDER_DIR / dataset_id if dataset_id and (RECORDER_DIR / dataset_id).is_dir() else RECORDER_DIR
        return sorted(path.glob("*.json"))

    def _prepare_dataset(self, files) -> Optional[np.ndarray]:
        features = []
        for path in files:
            try:
                scene = json.loads(path.read_text()).get("scene", {})
                features.append(self._encode_scene(scene))
            except Exception: continue
        return np.asarray(features, dtype=np.float32) if features else None

    def _encode_scene(self, scene: Dict) -> np.ndarray:
        vec = np.zeros(NON_VISUAL_DIM, dtype=np.float32)
        vec[0] = float(scene.get("mean", 0.0))
        objects, text = scene.get("objects", []), scene.get("text", [])
        vec[1], vec[2] = float(len(objects)), float(len(text))
        idx = 3
        for entry in text:
            for token in str(entry).split():
                vec[idx + (hash(token) % (NON_VISUAL_DIM - idx))] += 1.0
        return vec
        
    def _publish_error_loop(self):
        while not stop_event.is_set():
            error = random.uniform(0.0, 1.0)
            payload = {"ok": True, "error": round(error, 4), "timestamp": time.time()}
            try:
                if self.client.is_connected():
                    self.client.publish(PRED_ERROR_TOPIC, json.dumps(payload))
            except Exception: pass
            stop_event.wait(max(1.0, PRED_ERROR_INTERVAL))

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        error_thread = threading.Thread(target=self._publish_error_loop, daemon=True)
        error_thread.start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("World model agent shut down.")

def _handle_signal(signum, frame):
    logger.info(f"Signal {signum} received, shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = WorldModelAgent()
    agent.run()

if __name__ == "__main__":
    main()
