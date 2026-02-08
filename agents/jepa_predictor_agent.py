#!/usr/bin/env python3
"""
JEPA Predictor Agent
- Subscribes to vision/embeddings (z_t) and act/result (a_t)
- Predicts next latent state z_{t+1}
- Publishes prediction error to world_model/pred_error
- Performs online learning to adapt to game dynamics
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=os.getenv("JEPA_LOG_LEVEL", "INFO"), format="[jepa_predictor] %(message)s")
logger = logging.getLogger("jepa_predictor")

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
EMBED_TOPIC = os.getenv("VISION_EMBEDDINGS_TOPIC", "vision/embeddings")
ACT_RESULT_TOPIC = os.getenv("ACT_RESULT_TOPIC", "act/result")
PRED_ERROR_TOPIC = os.getenv("PRED_ERROR_TOPIC", "world_model/pred_error")
MODEL_PATH = Path(os.getenv("JEPA_PREDICTOR_PATH", "/mnt/ssd/models/jepa/predictor.pt"))

EMBED_DIM = int(os.getenv("VL_JEPA_EMBED_DIM", "512"))
ACTION_DIM = 32  # Latent action dimension
LR = float(os.getenv("JEPA_LR", "0.0003"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ONLINE_TRAINING = os.getenv("JEPA_ONLINE_TRAINING", "1") != "0"
JEPA_USE_FP16 = os.getenv("JEPA_USE_FP16", "0") != "0"
JEPA_GRAD_CLIP_NORM = float(os.getenv("JEPA_GRAD_CLIP_NORM", "1.0"))
STARTUP_JITTER_SEC = float(os.getenv("JEPA_STARTUP_JITTER_SEC", "0"))
CLEANUP_EVERY = int(os.getenv("JEPA_CUDA_CLEANUP_EVERY", "0"))
PYTORCH_ALLOC_CONF = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")

# --- Models ---

class ActionEncoder(nn.Module):
    """Maps discrete and continuous action components to a fixed vector."""
    def __init__(self, out_dim=32):
        super().__init__()
        self.action_types = {
            "wait": 0, "mouse_move": 1, "click_primary": 2, 
            "click_secondary": 3, "key_press": 4, "mouse_hold": 5, "mouse_release": 6
        }
        self.type_embed = nn.Embedding(len(self.action_types) + 1, 16)
        self.params_proj = nn.Linear(4, 16) # dx, dy, tx, ty
        self.out_proj = nn.Linear(32, out_dim)

    def forward(self, action_dict: dict):
        a_type = str(action_dict.get("action", "wait")).lower()
        type_idx = self.action_types.get(a_type, len(self.action_types))
        type_t = torch.tensor([type_idx], device=DEVICE)
        type_feat = self.type_embed(type_t)

        # Extract continuous params
        dx = float(action_dict.get("dx", 0)) / 100.0
        dy = float(action_dict.get("dy", 0)) / 100.0
        target = action_dict.get("target_norm") or [0.5, 0.5]
        tx, ty = float(target[0]), float(target[1])
        params_dtype = self.params_proj.weight.dtype
        params_t = torch.tensor([[dx, dy, tx, ty]], device=DEVICE, dtype=params_dtype)
        params_feat = self.params_proj(params_t)

        return self.out_proj(torch.cat([type_feat, params_feat], dim=-1))

class LatentPredictor(nn.Module):
    """JEPA-style predictor: (z_t, a_t) -> z_{t+1}."""
    def __init__(self, embed_dim=512, action_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, embed_dim)
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))

# --- Agent ---

class JepaPredictorAgent:
    @staticmethod
    def _state_dict_is_finite(state_dict: Dict[str, torch.Tensor]) -> bool:
        for value in state_dict.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False
        return True

    def __init__(self):
        if PYTORCH_ALLOC_CONF:
            logger.info("PYTORCH_CUDA_ALLOC_CONF=%s", PYTORCH_ALLOC_CONF)
        if STARTUP_JITTER_SEC > 0:
            delay = random.uniform(1.0, STARTUP_JITTER_SEC)
            logger.info("Startup jitter: sleeping %.2fs", delay)
            time.sleep(delay)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="jepa_predictor")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.action_encoder = None
        self.predictor = None
        self.optimizer = None
        self._init_models()
        self.criterion = nn.MSELoss()

        if MODEL_PATH.exists():
            try:
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                predictor_state = state_dict.get("predictor", {})
                action_encoder_state = state_dict.get("action_encoder", {})
                if (
                    isinstance(predictor_state, dict)
                    and isinstance(action_encoder_state, dict)
                    and self._state_dict_is_finite(predictor_state)
                    and self._state_dict_is_finite(action_encoder_state)
                ):
                    self.predictor.load_state_dict(predictor_state)
                    self.action_encoder.load_state_dict(action_encoder_state)
                    logger.info("Loaded weights from %s", MODEL_PATH)
                else:
                    self._quarantine_bad_checkpoint("non-finite checkpoint tensors")
            except Exception as e:
                logger.warning("Failed to load weights: %s", e)
                self._quarantine_bad_checkpoint(f"load error: {e}")

        self.last_z: Optional[torch.Tensor] = None
        self.last_action: Optional[dict] = None
        self.train_steps = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def _init_models(self):
        self.action_encoder = ActionEncoder(out_dim=ACTION_DIM).to(DEVICE)
        self.predictor = LatentPredictor(embed_dim=EMBED_DIM, action_dim=ACTION_DIM).to(DEVICE)
        if DEVICE.type == "cuda" and JEPA_USE_FP16:
            self.action_encoder.half()
            self.predictor.half()
            logger.info("JEPA Predictor optimized with FP16 (Half Precision)")
        elif DEVICE.type == "cuda":
            logger.info("JEPA Predictor running on CUDA with FP32 for stability")
        self.optimizer = optim.Adam(
            list(self.predictor.parameters()) + list(self.action_encoder.parameters()),
            lr=LR,
        )

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.predictor.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _modules_are_finite(self) -> bool:
        for module in (self.action_encoder, self.predictor):
            for param in module.parameters():
                if not torch.isfinite(param).all():
                    return False
        return True

    def _quarantine_bad_checkpoint(self, reason: str):
        logger.warning("Ignoring JEPA checkpoint (%s): %s", reason, MODEL_PATH)
        try:
            bad_path = MODEL_PATH.with_suffix(f".bad.{int(time.time())}.pt")
            MODEL_PATH.rename(bad_path)
            logger.warning("Moved bad checkpoint to %s", bad_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to quarantine bad checkpoint: %s", exc)

    def _recover_models(self, reason: str):
        logger.warning("Recovering JEPA models: %s", reason)
        self._init_models()

    def on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe([(EMBED_TOPIC, 0), (ACT_RESULT_TOPIC, 0)])
            logger.info("Subscribed to %s and %s", EMBED_TOPIC, ACT_RESULT_TOPIC)
        else:
            logger.error("Connect failed: rc=%s", rc)

    def on_message(self, _client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception: return

        with self.lock:
            if msg.topic == EMBED_TOPIC:
                self.handle_embedding(data)
            elif msg.topic == ACT_RESULT_TOPIC:
                self.handle_action(data)

    def handle_action(self, data: dict):
        # We look for 'applied' or 'action' in the payload
        action = data.get("applied") or data.get("action")
        if isinstance(action, dict):
            self.last_action = action

    def handle_embedding(self, data: dict):
        z_raw = data.get("embedding")
        if not z_raw or len(z_raw) != EMBED_DIM:
            return
        try:
            if not np.isfinite(np.asarray(z_raw, dtype=np.float32)).all():
                logger.warning("Skipping non-finite embedding payload")
                return
        except Exception:  # noqa: BLE001
            return
        
        z_curr = torch.tensor([z_raw], device=DEVICE, dtype=self._model_dtype())
        
        # If we have z_t and a_t, we can calculate prediction error for z_{t+1}
        if self.last_z is not None and self.last_action is not None:
            self.process_transition(self.last_z, self.last_action, z_curr)
        
        self.last_z = z_curr.detach()

    def process_transition(self, z_t, a_dict, z_next):
        model_dtype = self._model_dtype()
        z_t = z_t.to(dtype=model_dtype)
        z_next = z_next.to(dtype=model_dtype)
        # 1. Encode action
        with torch.no_grad():
            a_t = self.action_encoder(a_dict)
        
        # 2. Predict next latent
        self.predictor.eval()
        with torch.no_grad():
            z_pred = self.predictor(z_t, a_t)
            error_tensor = self.criterion(z_pred, z_next)
            if not torch.isfinite(error_tensor):
                logger.warning("Skipping non-finite pred_error; triggering recovery")
                self._recover_models("non-finite pred_error")
                return
            error = float(error_tensor.item())
        
        # 3. Publish error
        payload = {
            "ok": True, 
            "error": round(error, 6),
            "pred_error": round(error, 6),
            "timestamp": time.time(),
            "source": "jepa_predictor"
        }
        self.client.publish(PRED_ERROR_TOPIC, json.dumps(payload))
        
        # 4. Online training
        if ONLINE_TRAINING:
            self.predictor.train()
            self.action_encoder.train()
            self.optimizer.zero_grad()
            a_t_train = self.action_encoder(a_dict)
            z_pred_train = self.predictor(z_t, a_t_train)
            loss = self.criterion(z_pred_train, z_next.detach())
            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite training loss; triggering recovery")
                self._recover_models("non-finite training loss")
                return
            loss.backward()
            if JEPA_GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.predictor.parameters()) + list(self.action_encoder.parameters()),
                    JEPA_GRAD_CLIP_NORM,
                )
            self.optimizer.step()
            if not self._modules_are_finite():
                self._recover_models("non-finite model weights after optimizer step")
                return

            self.train_steps += 1
            if CLEANUP_EVERY > 0 and self.train_steps % CLEANUP_EVERY == 0:
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    def save_loop(self):
        while not self.stop_event.is_set():
            time.sleep(60) # Save every minute
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "predictor": self.predictor.state_dict(),
                "action_encoder": self.action_encoder.state_dict()
            }, MODEL_PATH)
            logger.debug("Checkpoint saved to %s", MODEL_PATH)

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        save_thread = threading.Thread(target=self.save_loop, daemon=True)
        save_thread.start()
        
        def handle_signal(signum, _frame):
            logger.info("Signal received, shutting down...")
            self.stop_event.set()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        self.stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    agent = JepaPredictorAgent()
    agent.run()
