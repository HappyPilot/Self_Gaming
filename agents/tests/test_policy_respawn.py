import time
import unittest

import agents.policy_agent as policy
from agents.policy_agent import PolicyAgent


class PolicyRespawnTest(unittest.TestCase):
    def setUp(self):
        self._old_require_embeddings = policy.POLICY_REQUIRE_EMBEDDINGS
        self._old_intent_enabled = policy.POLICY_INTENT_ENABLED
        self._old_scene_max_age = policy.POLICY_SCENE_MAX_AGE
        self._old_click_to_move = policy.POLICY_CLICK_TO_MOVE
        self._old_learning_stage = policy.LEARNING_STAGE
        self._old_hot_reload = policy.HOT_RELOAD_ENABLED

        policy.POLICY_REQUIRE_EMBEDDINGS = False
        policy.POLICY_INTENT_ENABLED = True
        policy.POLICY_SCENE_MAX_AGE = 30.0
        policy.POLICY_CLICK_TO_MOVE = True
        policy.LEARNING_STAGE = 1
        policy.HOT_RELOAD_ENABLED = False

    def tearDown(self):
        policy.POLICY_REQUIRE_EMBEDDINGS = self._old_require_embeddings
        policy.POLICY_INTENT_ENABLED = self._old_intent_enabled
        policy.POLICY_SCENE_MAX_AGE = self._old_scene_max_age
        policy.POLICY_CLICK_TO_MOVE = self._old_click_to_move
        policy.LEARNING_STAGE = self._old_learning_stage
        policy.HOT_RELOAD_ENABLED = self._old_hot_reload

    def test_respawn_action_overrides_recover_wait(self):
        agent = PolicyAgent()
        agent.game_keywords = set()
        now = time.time()
        state = {
            "ok": True,
            "event": "scene_update",
            "timestamp": now,
            "mean": 0.2,
            "flags": {"in_game": True, "death": True},
            "text": ["Resurrect at checkpoint"],
            "embeddings": [0.0] * 8,
            "embeddings_ts": now,
        }
        action = agent._policy_from_observation(state)
        self.assertIsNotNone(action, "expected respawn action")
        self.assertNotEqual(action.get("label"), "wait")

    def test_respawn_text_does_not_trigger_on_potion_tooltip(self):
        agent = PolicyAgent()
        tooltip = [
            "RECOVERS70LIFEOVER3SECONDS",
            "CONSUMES 7 OF 21 CHARGES ON USE",
            "CURRENTLY HAS21CHARGE",
        ]
        self.assertFalse(agent._scene_has_respawn_text(tooltip))

    def test_respawn_text_does_not_trigger_on_multiline_tooltip_blob(self):
        agent = PolicyAgent()
        tooltip_blob = (
            "RECOVERS70LIFEOVER3SECONDS\n"
            "CONSUMES 7 OF 21 CHARGES ON USE\n"
            "CURRENTLY HAS21CHARGE"
        )
        self.assertFalse(agent._scene_has_respawn_text([tooltip_blob]))

    def test_respawn_text_still_triggers_on_real_prompt(self):
        agent = PolicyAgent()
        self.assertTrue(agent._scene_has_respawn_text(["Resurrect at checkpoint"]))

    def test_respawn_text_does_not_trigger_on_stat_resource_line(self):
        agent = PolicyAgent()
        agent.respawn_keywords = {
            policy._normalize_phrase(text)
            for text in [
                "respawn",
                "revive",
                "resurrect",
                "resurrect at checkpoint",
                "resurrect in town",
                "restart",
                "try again",
                "game over",
                "you died",
                "continue",
                "loading",
                "checkpoint",
            ]
        }
        self.assertFalse(agent._scene_has_respawn_text(["stat_1:81/81"]))


if __name__ == "__main__":
    unittest.main()
