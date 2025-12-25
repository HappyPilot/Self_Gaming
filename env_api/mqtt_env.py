"""MQTT-backed EnvAdapter implementation."""
from __future__ import annotations

import json
import os
import queue
import time
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt
except Exception:  # noqa: BLE001
    mqtt = None

from env_api.adapter import Action, EnvAdapter, Observation, StepResult

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "30"))

OBS_TOPIC = os.getenv("ENV_OBS_TOPIC") or os.getenv("PERCEPTION_TOPIC", "vision/observation")
REWARD_TOPIC = os.getenv("ENV_REWARD_TOPIC") or os.getenv("REWARD_TOPIC", "train/reward")
ACTION_TOPIC = os.getenv("ENV_ACTION_TOPIC") or os.getenv("ACT_CMD_TOPIC", "act/cmd")

OBS_QUEUE_MAX = int(os.getenv("ENV_OBS_QUEUE_MAX", "200"))
REWARD_QUEUE_MAX = int(os.getenv("ENV_REWARD_QUEUE_MAX", "200"))
STEP_TIMEOUT_SEC = float(os.getenv("ENV_STEP_TIMEOUT_SEC", "1.0"))
OBS_TIMEOUT_SEC = float(os.getenv("ENV_OBS_TIMEOUT_SEC", "2.0"))
HEALTH_STALE_SEC = float(os.getenv("ENV_HEALTH_STALE_SEC", "5.0"))


def _queue_put(q: queue.Queue, item: object) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _coerce_action(action: Dict[str, Any] | Action) -> Dict[str, Any]:
    if isinstance(action, Action):
        payload = dict(action.payload)
        if action.timestamp is not None:
            payload.setdefault("timestamp", action.timestamp)
        return payload
    return dict(action)


class MqttEnvAdapter(EnvAdapter):
    """EnvAdapter that uses MQTT topics for observation and actions."""

    def __init__(self) -> None:
        if mqtt is None:
            raise RuntimeError("paho-mqtt is required for MqttEnvAdapter")
        self._obs_queue: queue.Queue[Observation] = queue.Queue(maxsize=OBS_QUEUE_MAX)
        self._reward_queue: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=REWARD_QUEUE_MAX)
        self._latest_obs: Optional[Observation] = None
        self._latest_reward: Optional[tuple[float, float]] = None
        self._last_obs_ts: Optional[float] = None
        self._connected = False
        self.client = mqtt.Client(client_id="env_adapter", protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
        self.client.loop_start()

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            topics = [(OBS_TOPIC, 0)]
            if REWARD_TOPIC:
                topics.append((REWARD_TOPIC, 0))
            client.subscribe(topics)
            self._connected = True
        else:
            self._connected = False

    def _on_disconnect(self, _client, _userdata, _rc):
        self._connected = False

    def _build_observation(self, payload: dict, topic: str) -> Observation:
        ts = float(payload.get("timestamp", time.time()))
        frame_id = payload.get("frame_id")
        try:
            frame_id = int(frame_id) if frame_id is not None else None
        except (TypeError, ValueError):
            frame_id = None
        return Observation(timestamp=ts, frame_id=frame_id, payload=payload, topic=topic)

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        if msg.topic == OBS_TOPIC:
            obs = self._build_observation(payload, msg.topic)
            self._latest_obs = obs
            self._last_obs_ts = obs.timestamp
            _queue_put(self._obs_queue, obs)
        elif msg.topic == REWARD_TOPIC:
            value = payload.get("reward")
            try:
                reward = float(value)
            except (TypeError, ValueError):
                return
            ts = float(payload.get("timestamp", time.time()))
            self._latest_reward = (reward, ts)
            _queue_put(self._reward_queue, (reward, ts))

    def _wait_for_obs(self, after_ts: Optional[float], timeout_sec: float) -> Optional[Observation]:
        deadline = time.time() + max(0.0, timeout_sec)
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                obs = self._obs_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if after_ts is None or obs.timestamp > after_ts:
                return obs
        return None

    def _reward_after(self, after_ts: float) -> Optional[float]:
        if self._latest_reward and self._latest_reward[1] >= after_ts:
            return self._latest_reward[0]
        return None

    def reset(self) -> Observation:
        obs = self.get_observation(timeout_sec=OBS_TIMEOUT_SEC)
        if obs is None:
            raise RuntimeError("No observation available for reset()")
        return obs

    def step(self, action: Dict[str, Any] | Action) -> StepResult:
        action_payload = _coerce_action(action)
        action_ts = float(action_payload.get("timestamp", time.time()))
        try:
            self.client.publish(ACTION_TOPIC, json.dumps(action_payload))
        except Exception:
            return StepResult(None, None, False, info={"publish_failed": True}, health=self.health_check())
        obs = self._wait_for_obs(self._last_obs_ts, STEP_TIMEOUT_SEC)
        reward = self._reward_after(action_ts)
        info = {"action_ts": action_ts}
        if obs is None:
            info["timeout"] = True
        return StepResult(obs, reward, False, info=info, health=self.health_check())

    def get_observation(self, timeout_sec: Optional[float] = None) -> Optional[Observation]:
        timeout = OBS_TIMEOUT_SEC if timeout_sec is None else timeout_sec
        obs = self._wait_for_obs(None, timeout)
        if obs is not None:
            self._latest_obs = obs
            self._last_obs_ts = obs.timestamp
        return obs

    def health_check(self) -> Dict[str, Any]:
        now = time.time()
        last_obs_age = None
        if self._last_obs_ts is not None:
            last_obs_age = round(now - self._last_obs_ts, 3)
        ok = self._connected and (last_obs_age is None or last_obs_age <= HEALTH_STALE_SEC)
        return {
            "ok": ok,
            "connected": self._connected,
            "last_obs_age_sec": last_obs_age,
            "obs_topic": OBS_TOPIC,
            "reward_topic": REWARD_TOPIC,
            "action_topic": ACTION_TOPIC,
        }

    def close(self) -> None:
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass
