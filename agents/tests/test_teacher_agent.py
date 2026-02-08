import json
import unittest

from agents.teacher_agent import TeacherAgent


class DummyLLM:
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


class DummyMQTT:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload):
        self.messages.append((topic, json.loads(payload)))


class DummyMem:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def query(self, payload, timeout=1.0):
        self.calls.append(payload)
        mode = payload.get("mode")
        return self.responses.get(mode, {"value": []})


class TeacherAgentTest(unittest.TestCase):
    def test_generate_action_uses_llm_and_publishes(self):
        mem = DummyMem({})
        agent = TeacherAgent(mqtt_client=DummyMQTT(), llm_client=DummyLLM(), mem_client=mem)
        agent.scene = {"ok": True, "mean": 42.0, "text": ["Search", "Box"]}
        agent.snapshot = "abcdef" * 10
        agent.actions.extend(["pressed_tab", "typed_weather"])

        agent._generate_action(agent.client)

        self.assertTrue(agent.client.messages, "Teacher agent did not publish an action")
        topic, payload = agent.client.messages[0]
        self.assertIn("teacher/action", topic)
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
        topic, payload = agent.client.messages[0]
        self.assertEqual(payload["rules_used"], 1)
        self.assertEqual(payload["recent_critical_used"], 1)

    def test_playbook_is_included_in_scene_summary(self):
        mem = DummyMem({})
        agent = TeacherAgent(mqtt_client=DummyMQTT(), llm_client=DummyLLM(), mem_client=mem)
        agent.scene = {"ok": True, "text": ["Start", "Menu"]}
        agent.snapshot = "abcdef" * 10

        agent._generate_action(agent.client)

        scene_summary = agent.llm.summary_args[0]
        self.assertIn("Playbook:", scene_summary)


if __name__ == "__main__":
    unittest.main()
