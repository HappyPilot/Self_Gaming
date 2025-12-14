#!/usr/bin/env python3
"""End-to-end smoke test for the multi-agent stack.

The test drives a minimal teach → train → eval loop over MQTT:
1. Seed the recorder with synthetic scene/action pairs.
2. Ask teach_agent to plan a training job.
3. Wait for train_manager_agent to finish that job.
4. Trigger eval_agent on the resulting plan id and assert success.

The script exits with code 0 on success and non-zero on failure.
"""
import argparse
import json
import queue
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import paho.mqtt.client as mqtt

TopicPredicate = Callable[[str, Dict[str, object]], bool]


@dataclass
class SmokeConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    timeout: float = 20.0
    recorder_topic: str = "recorder/status"
    scene_topic: str = "scene/state"
    act_topic: str = "act/request"
    teach_topic: str = "teach/request"
    train_job_topic: str = "train/jobs"
    train_status_topic: str = "train/status"
    eval_request_topic: str = "eval/request"
    eval_result_topics: Tuple[str, ...] = ("eval/result", "eval/report")


class SmokeTest:
    def __init__(self, config: SmokeConfig):
        self.cfg = config
        self.client = mqtt.Client(client_id="smoke_test", protocol=mqtt.MQTTv311)
        self.connected = False
        self.events: Dict[str, list] = defaultdict(list)
        self.cursor: Dict[str, int] = defaultdict(int)
        self.queue: "queue.Queue[str]" = queue.Queue()

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # MQTT callbacks -----------------------------------------------------
    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            subs = {
                (self.cfg.recorder_topic, 0),
                (self.cfg.train_job_topic, 0),
                (self.cfg.train_status_topic, 0),
            }
            for topic in self.cfg.eval_result_topics:
                subs.add((topic, 0))
            client.subscribe(list(subs))
            self.connected = True
        else:
            raise RuntimeError(f"MQTT connection failed with rc={rc}")

    def _on_message(self, _client, _userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", "ignore")
            data = json.loads(payload)
            if not isinstance(data, dict):
                data = {"value": data}
        except Exception:
            data = {"value": msg.payload.decode("utf-8", "ignore")}
        self.events[msg.topic].append(data)
        self.queue.put(msg.topic)

    # Core helpers -------------------------------------------------------
    def connect(self):
        self.client.connect(self.cfg.host, self.cfg.port, 30)
        self.client.loop_start()
        t0 = time.time()
        while not self.connected and time.time() - t0 < 5:
            time.sleep(0.05)
        if not self.connected:
            raise TimeoutError("Timed out waiting for MQTT connection")

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()

    def wait_for(self, topics: Iterable[str], predicate: TopicPredicate, timeout: float):
        deadline = time.time() + timeout
        topics = tuple(topics)
        while time.time() < deadline:
            for topic in topics:
                cursor = self.cursor[topic]
                events = self.events.get(topic, [])
                while cursor < len(events):
                    data = events[cursor]
                    cursor += 1
                    if predicate(topic, data):
                        self.cursor[topic] = cursor
                        return topic, data
                self.cursor[topic] = cursor
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                self.queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
        raise TimeoutError(f"Timed out waiting for topics {topics}")

    # Test flow ----------------------------------------------------------
    def ensure_recorder_ready(self):
        try:
            self.wait_for(
                [self.cfg.recorder_topic],
                lambda _t, data: data.get("event") == "recorder_ready",
                timeout=2.0,
            )
        except TimeoutError:
            # Recorder readiness is helpful but not fatal; proceed anyway.
            pass

    def seed_recorder(self):
        samples = [
            (0.25, ["menu"], {"action": "press_enter"}),
            (0.35, ["dialog"], {"action": "press_enter"}),
            (0.45, ["button"], {"action": "mouse_move", "dx": 100, "dy": -80}),
            (0.55, ["field"], {"action": "mouse_move", "dx": -120, "dy": 90}),
        ]
        for idx, (mean, text, action_payload) in enumerate(samples):
            scene_payload = {
                "ok": True,
                "event": "scene_update",
                "mean": mean,
                "trend": [mean - 0.05, mean],
                "text": text,
                "timestamp": time.time(),
                "source": "smoke_test",
                "step": idx,
            }
            self.client.publish(self.cfg.scene_topic, json.dumps(scene_payload), qos=0)
            time.sleep(0.05)
            publish_payload = {"ok": True, "source": "smoke_test", "step": idx, **action_payload}
            self.client.publish(self.cfg.act_topic, json.dumps(publish_payload), qos=0)
            time.sleep(0.05)
        # Give recorder a moment to flush to disk.
        time.sleep(0.5)

    def trigger_training(self) -> str:
        # Prime teach_agent with a fresh scene snapshot.
        scene_payload = {
            "ok": True,
            "event": "scene_update",
            "mean": 0.6,
            "trend": [0.5, 0.6],
            "text": ["ready"],
            "timestamp": time.time(),
            "source": "smoke_test",
            "step": "train",
        }
        self.client.publish(self.cfg.scene_topic, json.dumps(scene_payload), qos=0)

        plan_request = {"cmd": "plan", "source": "smoke_test"}
        self.client.publish(self.cfg.teach_topic, json.dumps(plan_request), qos=0)

        _topic, job_payload = self.wait_for(
            [self.cfg.train_job_topic],
            lambda _t, data: data.get("event") == "train_job_created",
            timeout=5.0,
        )
        job_id = job_payload.get("job_id")
        if not job_id:
            raise RuntimeError("Teach agent did not provide a job_id")

        # Wait for the job to start.
        self.wait_for(
            [self.cfg.train_status_topic],
            lambda _t, data: data.get("job_id") == job_id and data.get("event") == "job_started",
            timeout=5.0,
        )

        # Wait for job completion (finished or failed).
        _topic, completion = self.wait_for(
            [self.cfg.train_status_topic],
            lambda _t, data: data.get("job_id") == job_id and data.get("event") in {"job_finished", "job_failed"},
            timeout=self.cfg.timeout,
        )
        if completion.get("event") == "job_failed":
            raise RuntimeError(f"Training job failed: {completion}")
        return job_id

    def trigger_eval(self, job_id: str) -> Dict:
        request = {"plan_id": job_id, "source": "smoke_test"}
        self.client.publish(self.cfg.eval_request_topic, json.dumps(request), qos=0)

        topic, result = self.wait_for(
            self.cfg.eval_result_topics,
            lambda _t, data: data.get("plan") == job_id,
            timeout=5.0,
        )
        if not result.get("ok"):
            raise RuntimeError(f"Eval agent reported failure on {topic}: {result}")
        return result


def parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(description="Smoke test the multi-agent training loop")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--scene-topic", default="scene/state")
    parser.add_argument("--act-topic", default="act/request")
    parser.add_argument("--teach-topic", default="teach/request")
    parser.add_argument("--eval-request-topic", default="eval/request")
    parser.add_argument(
        "--eval-result-topics",
        default="eval/result,eval/report",
        help="Comma-separated set of topics that carry evaluation results",
    )
    args = parser.parse_args()
    eval_result_topics = tuple(t.strip() for t in args.eval_result_topics.split(",") if t.strip())
    return SmokeConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        scene_topic=args.scene_topic,
        act_topic=args.act_topic,
        teach_topic=args.teach_topic,
        eval_request_topic=args.eval_request_topic,
        eval_result_topics=eval_result_topics,
    )


def main() -> int:
    cfg = parse_args()
    test = SmokeTest(cfg)
    try:
        test.connect()
        test.ensure_recorder_ready()
        test.seed_recorder()
        job_id = test.trigger_training()
        result = test.trigger_eval(job_id)
        score = result.get("score")
        print(json.dumps({"ok": True, "job_id": job_id, "score": score, "topic": result.get("event") or "eval"}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    finally:
        test.close()


if __name__ == "__main__":
    sys.exit(main())
