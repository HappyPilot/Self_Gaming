import json
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional

import paho.mqtt.client as mqtt
import logging

try:
    import requests
except ImportError:  # pragma: no cover - optional for offline tests
    requests = None

try:
    from .mem_rpc import MemRPC  # type: ignore
except ImportError:  # pragma: no cover - script invocation fallback
    from mem_rpc import MemRPC
try:
    from .llm_gate import acquire_gate, blocked_reason, release_gate
except ImportError:  # pragma: no cover - script invocation fallback
    from llm_gate import acquire_gate, blocked_reason, release_gate

logging.basicConfig(level=os.getenv("RESEARCH_LOG_LEVEL", "INFO"))
logger = logging.getLogger("research_agent")

# --- Constants ---
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
RESEARCH_TOPIC = os.getenv("RESEARCH_TOPIC", "research/events")
GOAL_TOPIC = os.getenv("GOALS_TOPIC", "goals/high_level")
TRAIN_JOB_TOPIC = os.getenv("TRAIN_JOB_TOPIC", "train/jobs")
DOC_DIR = Path(os.getenv("RESEARCH_DOC_DIR", "/mnt/ssd/research/cache"))
INTERVAL = float(os.getenv("RESEARCH_INTERVAL", "120"))
SIM_CMD_TOPIC = os.getenv("SIM_CMD_TOPIC", "sim_core/cmd")
DEFAULT_MODE = os.getenv("RESEARCH_TRAIN_MODE", "ppo_baseline")
VISION_CONFIG_TOPIC = os.getenv("VISION_CONFIG_TOPIC", "vision/config")
VISION_MODES = [m.strip().lower() for m in os.getenv("RESEARCH_VISION_MODES", "low,medium,high").split(",") if m.strip()]
MEM_QUERY_TOPIC = os.getenv("MEM_QUERY_TOPIC", "mem/query")
MEM_REPLY_TOPIC = os.getenv("MEM_REPLY_TOPIC", "mem/reply")
MEM_RESPONSE_TOPIC = os.getenv("MEM_RESPONSE_TOPIC", "mem/response")
MEM_STORE_TOPIC = os.getenv("MEM_STORE_TOPIC", "mem/store")

REFLECTION_ENABLED = os.getenv("RESEARCH_REFLECTION_ENABLED", "1") not in {"0", "false", "False"}
REFLECTION_INTERVAL = float(os.getenv("RESEARCH_REFLECTION_INTERVAL", "600"))
REFLECTION_SCOPES = [s.strip() for s in os.getenv("RESEARCH_REFLECTION_SCOPES", "death_dialog,early_game_farm,generic_ui").split(",") if s.strip()]
REFLECTION_PINNED_LIMIT = int(os.getenv("RESEARCH_REFLECTION_PINNED", "5"))
REFLECTION_RULE_MAX = int(os.getenv("RESEARCH_REFLECTION_RULE_MAX", "5"))
REFLECTION_ENDPOINT = os.getenv("RESEARCH_LLM_ENDPOINT", os.getenv("TEACHER_LOCAL_ENDPOINT"))
REFLECTION_MODEL = os.getenv("RESEARCH_LLM_MODEL", "gpt-4o-mini")

stop_event = threading.Event()


def _as_int(code) -> int:
    try:
        if hasattr(code, "value"):
            return int(code.value)
        return int(code)
    except (TypeError, ValueError):
        return 0


class ResearchAgent:
    def __init__(self) -> None:
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id="research_agent")
        self.client.on_connect = self.on_connect
        self.docs: List[str] = []
        self._load_docs()
        self.mem_rpc: Optional[MemRPC] = None
        self._reflection_enabled = REFLECTION_ENABLED and REFLECTION_SCOPES
        if self._reflection_enabled and (requests is None or not REFLECTION_ENDPOINT):
            logger.warning("Disabling reflection: requests installed=%s endpoint=%s", bool(requests), bool(REFLECTION_ENDPOINT))
            self._reflection_enabled = False
        self._next_reflection = time.time() + REFLECTION_INTERVAL
        self._reflection_index = 0
        if self._reflection_enabled:
            self.mem_rpc = MemRPC(
                host=MQTT_HOST,
                port=MQTT_PORT,
                query_topic=MEM_QUERY_TOPIC,
                reply_topic=MEM_RESPONSE_TOPIC,
            )

    def _load_docs(self):
        if DOC_DIR.exists():
            for path in DOC_DIR.glob("**/*.txt"):
                try:
                    self.docs.append(path.read_text(encoding="utf-8"))
                except Exception:
                    continue

    def on_connect(self, client, _userdata, _flags, rc):
        if _as_int(rc) == 0:
            client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "research_ready"}))
            # Start the loop in a background thread only once on successful connect.
            if not any(t.name == "_loop" for t in threading.enumerate()):
                 threading.Thread(target=self._loop, name="_loop", daemon=True).start()
        else:
            logger.error("Research agent failed to connect: rc=%s", _as_int(rc))
            client.publish(RESEARCH_TOPIC, json.dumps({"ok": False, "event": "connect_failed", "code": _as_int(rc)}))

    def _loop(self):
        while not stop_event.is_set():
            self.publish_goal()
            self.launch_sim_experiment()
            self._maybe_reflect()
            stop_event.wait(INTERVAL)

    def publish_goal(self):
        idea = random.choice(self.docs) if self.docs else "Explore new map layout"
        goal = {
            "ok": True,
            "goal_id": f"research_{int(time.time())}",
            "goal_type": "research",
            "reasoning": idea[:256],
            "source": "research_agent",
            "timestamp": time.time(),
        }
        self.client.publish(GOAL_TOPIC, json.dumps(goal))
        self.client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "goal_proposed", "goal_id": goal["goal_id"]}))

    def launch_sim_experiment(self):
        job_id = f"research_sim_{int(time.time())}"
        mode = DEFAULT_MODE
        target = None
        if random.random() < 0.2:
            mode = "world_model_experiment"
            target = "world_model"
        payload = {
            "ok": True,
            "job_id": job_id,
            "mode": mode,
            "dataset": "sim_core",
        }
        if target:
            payload["target"] = target
        self.client.publish(TRAIN_JOB_TOPIC, json.dumps(payload))
        dr_level = random.choice(["low", "medium", "high"])
        if SIM_CMD_TOPIC:
            self.client.publish(SIM_CMD_TOPIC, json.dumps({"cmd": "set_dr", "level": dr_level}))
        if VISION_CONFIG_TOPIC and VISION_MODES:
            mode_choice = dr_level if dr_level in VISION_MODES else random.choice(VISION_MODES)
            cfg = {
                "mode": mode_choice,
                "reason": "research_experiment",
                "ttl_sec": 15,
            }
            self.client.publish(VISION_CONFIG_TOPIC, json.dumps(cfg))
        self.client.publish(RESEARCH_TOPIC, json.dumps({"ok": True, "event": "experiment_started", "job_id": job_id}))

    def _maybe_reflect(self):
        if not self._reflection_enabled or not self.mem_rpc:
            return
        now = time.time()
        if now < self._next_reflection:
            return
        reason = blocked_reason()
        if reason:
            logger.info("LLM blocked (%s); skipping reflection", reason)
            self._next_reflection = time.time() + REFLECTION_INTERVAL
            return
        scope = REFLECTION_SCOPES[self._reflection_index % len(REFLECTION_SCOPES)]
        self._reflection_index += 1
        try:
            created = self._run_reflection(scope)
            if created:
                self.client.publish(
                    RESEARCH_TOPIC,
                    json.dumps({
                        "ok": True,
                        "event": "reflection_rules_created",
                        "scope": scope,
                        "count": created,
                    }),
                )
        except Exception as exc:  # pragma: no cover - reflection errors shouldn't kill agent
            logger.error("Reflection failed for scope %s: %s", scope, exc)
            self.client.publish(
                RESEARCH_TOPIC,
                json.dumps({"ok": False, "event": "reflection_failed", "scope": scope, "error": str(exc)[:160]}),
            )
        finally:
            self._next_reflection = time.time() + REFLECTION_INTERVAL

    def _run_reflection(self, scope: str) -> int:
        payload = {"mode": "pinned", "limit": REFLECTION_PINNED_LIMIT}
        if scope:
            payload["scope"] = scope
        response = self.mem_rpc.query(payload, timeout=2.0)
        episodes = (response or {}).get("value") or []
        if not episodes:
            return 0
        summaries = []
        for ep in episodes:
            summary = ep.get("summary", "")
            score = ep.get("score")
            summaries.append(f"score={score} :: {summary[:160]}")
        rules = self._generate_rules(scope, summaries)
        if not rules:
            return 0
        created = 0
        for rule in rules[:REFLECTION_RULE_MAX]:
            rule_payload = {
                "op": "rule_insert",
                "value": {
                    "scope": rule.get("scope") or scope,
                    "text": rule.get("text") or rule.get("rule"),
                    "confidence": rule.get("confidence", 0.6),
                    "source": "reflection",
                },
            }
            if rule_payload["value"]["text"]:
                self.client.publish(MEM_STORE_TOPIC, json.dumps(rule_payload))
                created += 1
        if created:
            logger.info("Reflection added %s rules for scope %s", created, scope)
        return created

    def _generate_rules(self, scope: str, summaries: List[str]) -> List[dict]:
        if not REFLECTION_ENDPOINT or requests is None:
            raise RuntimeError("Reflection endpoint not configured or requests missing")
        acquired = acquire_gate("research_reflection", wait_s=0.0)
        if not acquired:
            return []
        prompt = (
            "You analyze gameplay episodes to produce concise rules for an automation agent.\n"
            f"Scope: {scope or 'generic'}\n"
            "Summaries:\n- " + "\n- ".join(summaries) + "\n"
            "Return a JSON list where each item has 'scope', 'text', and 'confidence' (0-1)."
        )
        payload = {
            "model": REFLECTION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a compact gameplay strategist."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        try:
            response = requests.post(REFLECTION_ENDPOINT, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
        finally:
            release_gate()
        content = ""
        if isinstance(data, dict):
            if data.get("choices"):
                content = data["choices"][0]["message"]["content"]
            elif data.get("message"):
                content = data["message"].get("content", "")
            else:
                content = json.dumps(data)
        else:
            content = str(data)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "rules" in parsed:
                parsed = parsed["rules"]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # Fallback: parse bullet lines "- scope: text (confidence=0.5)"
        rules: List[dict] = []
        for line in content.splitlines():
            if "-" not in line:
                continue
            text = line.strip("- ")
            if not text:
                continue
            conf = 0.6
            if "confidence" in text:
                try:
                    conf = float(text.split("confidence")[-1].strip(" =:)"))
                except Exception:
                    conf = 0.6
            rules.append({"scope": scope, "text": text, "confidence": conf})
        return rules

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, 30)
        self.client.loop_start()
        stop_event.wait()
        self.client.loop_stop()
        self.client.disconnect()


def _handle_signal(signum, frame):
    logger.info("Signal %s received, shutting down research agent.", signum)
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    agent = ResearchAgent()
    agent.run()


if __name__ == "__main__":
    main()
