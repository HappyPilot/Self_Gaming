import json
import unittest

from agents.teacher_agent import TeacherAgent


class DummyLLM:
    def summarize(self, scene_summary, snapshot_hint, recent_actions):
        self.summary_args = (scene_summary, snapshot_hint, recent_actions)
        return "The interface shows a search box."

    def propose_action(self, reasoning, scene_summary, recent_actions):
        self.action_args = (reasoning, scene_summary, recent_actions)
        return "Click the search box and type weather"


class DummyMQTT:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload):
        self.messages.append((topic, json.loads(payload)))


class TeacherAgentTest(unittest.TestCase):
    def test_generate_action_uses_llm_and_publishes(self):
        agent = TeacherAgent(mqtt_client=DummyMQTT(), llm_client=DummyLLM())
        agent.scene = {"ok": True, "mean": 42.0, "text": ["Search", "Box"]}
        agent.snapshot = "abcdef" * 10
        agent.actions.extend(["pressed_tab", "typed_weather"])

        # call private method directly to avoid threading complexity
        agent._generate_action(agent.client)

        self.assertTrue(agent.client.messages, "Teacher agent did not publish an action")
        topic, payload = agent.client.messages[0]
        self.assertIn("teacher/action", topic)
        self.assertTrue(payload["action"].lower().startswith("click"))
        self.assertIn("reasoning", payload)


if __name__ == "__main__":
    unittest.main()
