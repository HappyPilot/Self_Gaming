import json
import unittest

import agents.teacher_agent as teacher_mod
from agents.teacher_agent import TeacherAgent


class DummyLLM:
    def __init__(self):
        self.describe_calls = 0

    def summarize(self, scene_summary, snapshot_hint, recent_actions):
        self.summary_args = (scene_summary, snapshot_hint, recent_actions)
        return "The interface shows a search box."

    def propose_action(self, reasoning, scene_summary, recent_actions):
        self.action_args = (reasoning, scene_summary, recent_actions)
        return json.dumps(
            {
                "action": {"label": "click_primary", "target_norm": [0.5, 0.5], "target_label": "search box"},
                "reasoning": "Click the search box",
            }
        )

    def describe_environment(self, scene_text, objects):
        self.describe_calls += 1
        payload = {
            "game": "Path of Exile",
            "summary": "Action RPG with mouse and keyboard controls",
            "controls": ["Left click to move", "Right click to use skill"],
        }
        return json.dumps(payload)


class DummyMQTT:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload):
        self.messages.append((topic, json.loads(payload)))


class DummyMem:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def query(self, payload, timeout=1.0):
        self.calls.append(payload)
        mode = payload.get("mode")
        if mode:
            return self.responses.get(mode, {"value": []})
        key = payload.get("key")
        if key and key in self.responses:
            return {"value": self.responses[key]}
        return {"value": []}


class TeacherAgentTest(unittest.TestCase):
    def test_generate_action_uses_llm_and_publishes(self):
        mem = DummyMem({})
        agent = TeacherAgent(mqtt_client=DummyMQTT(), llm_client=DummyLLM(), mem_client=mem)
        agent.scene = {"ok": True, "mean": 42.0, "text": ["Search", "Box"]}
        agent.snapshot = "abcdef" * 10
        agent.actions.extend(["pressed_tab", "typed_weather"])

        agent._generate_action(agent.client)

        action = self._find_message(agent.client.messages, teacher_mod.TEACHER_TOPIC)
        self.assertIsNotNone(action, "Teacher agent did not publish an action")
        payload = action
        self.assertEqual(payload["action"]["label"], "click_primary")
        self.assertIn("reasoning", payload)

    def test_rules_and_recent_critical_in_prompt(self):
        mem = DummyMem(
            {
                "rules": {
                    "value": [
                        {
                            "rule_id": "r1",
                            "scope": "critical_dialog:death",
                            "text": "Move cursor to the bright button",
                        }
                    ]
                },
                "recent_critical": {
                    "value": [
                        {
                            "timestamp": 1000,
                            "delta": "hero_dead",
                            "episode_id": "sample_1.json",
                            "scope": "critical_dialog:death",
                        }
                    ]
                },
            }
        )
        agent = TeacherAgent(mqtt_client=DummyMQTT(), llm_client=DummyLLM(), mem_client=mem)
        agent.scene = {"ok": True, "flags": {"death": True}, "text": ["Resurrect"]}
        agent.snapshot = "abcdef" * 10

        agent._generate_action(agent.client)

        scene_summary = agent.llm.summary_args[0]
        self.assertIn("Move cursor to the bright button", scene_summary)
        self.assertIn("hero_dead", scene_summary)
        action = self._find_message(agent.client.messages, teacher_mod.TEACHER_TOPIC)
        self.assertIsNotNone(action)
        self.assertEqual(action["rules_used"], 1)
        self.assertEqual(action["recent_critical_used"], 1)

    def test_context_is_stored_and_reused(self):
        mem = DummyMem()
        client = DummyMQTT()
        agent = TeacherAgent(mqtt_client=client, llm_client=DummyLLM(), mem_client=mem)
        agent.scene = {
            "ok": True,
            "text": ["Path of Exile", "Health", "Mana"],
            "objects": [{"label": "hero"}, {"label": "SHOP"}],
        }
        agent._generate_action(agent.client)

        context_messages = [payload for topic, payload in client.messages if topic == teacher_mod.MEM_STORE_TOPIC]
        self.assertTrue(
            any(msg.get("op") == "set" and msg.get("key", "").startswith(teacher_mod.TEACHER_CONTEXT_KEY) for msg in context_messages),
            "Teacher agent did not persist context to mem",
        )
        self.assertGreater(agent.llm.describe_calls, 0)
        action = self._find_message(client.messages, teacher_mod.TEACHER_TOPIC)
        self.assertIsNotNone(action)
        self.assertEqual(action.get("context_game"), "Path of Exile")

    def _find_message(self, messages, topic):
        for current_topic, payload in messages:
            if current_topic == topic:
                return payload
        return None


if __name__ == "__main__":
    unittest.main()
