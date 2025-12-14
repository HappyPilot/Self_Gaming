import sys
import time
import types
import unittest


def _ensure_dummy_mqtt_module():
    if "paho.mqtt.client" in sys.modules:
        return
    client_mod = types.ModuleType("paho.mqtt.client")

    class _DummyClient:
        MQTTv311 = 4

        def __init__(self, *args, **kwargs):
            pass

        def loop_forever(self):
            return

        def connect(self, *args, **kwargs):
            return

        def subscribe(self, *args, **kwargs):
            return

        def publish(self, *args, **kwargs):
            return

    client_mod.Client = _DummyClient
    client_mod.MQTTv311 = 4
    mqtt_pkg = types.ModuleType("paho.mqtt")
    sys.modules.setdefault("paho", types.ModuleType("paho"))
    sys.modules["paho.mqtt"] = mqtt_pkg
    sys.modules["paho.mqtt.client"] = client_mod


def _ensure_dummy_torch_module():
    if "torch" in sys.modules:
        return
    torch_mod = types.ModuleType("torch")

    class _DummyTensor:
        def __init__(self, *args, **kwargs):
            self.args = args

        def to(self, *args, **kwargs):
            return self

        def unsqueeze(self, *_args, **_kwargs):
            return self

        def __getitem__(self, _key):
            return 0

    def _zeros(*args, **kwargs):
        return _DummyTensor(*args, **kwargs)

    def _device(name):
        return name

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _ArgMaxResult:
        def item(self):
            return 0

    def _argmax(*_args, **_kwargs):
        return _ArgMaxResult()

    torch_mod.device = _device
    torch_mod.float32 = float
    torch_mod.zeros = _zeros
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.argmax = _argmax
    torch_mod.load = lambda *args, **kwargs: {}

    nn_mod = types.ModuleType("torch.nn")

    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class _DummyLinear(_DummyModule):
        def load_state_dict(self, *args, **kwargs):
            return None

    nn_mod.Linear = _DummyLinear
    nn_mod.Module = _DummyModule
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod


_ensure_dummy_mqtt_module()
_ensure_dummy_torch_module()

from agents.policy_agent import PolicyAgent, compute_cursor_motion


class PolicyAgentTest(unittest.TestCase):
    def test_teacher_alpha_anneals_linearly(self):
        agent = PolicyAgent()
        agent.teacher_alpha_start = 1.0
        agent.teacher_decay_steps = 10
        agent.teacher_min_alpha = 0.0

        agent.steps = 0
        self.assertAlmostEqual(agent._current_alpha(), 1.0)
        agent.steps = 5
        self.assertAlmostEqual(agent._current_alpha(), 0.5)
        agent.steps = 10
        self.assertAlmostEqual(agent._current_alpha(), 0.0)
        agent.steps = 20
        self.assertAlmostEqual(agent._current_alpha(), 0.0)

    def test_compute_cursor_motion_includes_offsets(self):
        target_px, cursor_px, delta = compute_cursor_motion(
            x_norm=0.75,
            y_norm=0.25,
            cursor_x_norm=0.5,
            cursor_y_norm=0.5,
            width=1000,
            height=800,
            offset_x=10,
            offset_y=-20,
        )
        self.assertEqual(target_px, (760, 180))
        self.assertEqual(cursor_px, (510, 380))
        self.assertEqual(delta, (250, -200))

    def test_scene_allows_action_blocks_bright_without_keywords(self):
        agent = PolicyAgent()
        agent.game_keywords = {"poe"}
        state = {
            "timestamp": time.time(),
            "mean": 0.9,
            "text": ["desktop"],
        }
        allowed, reason = agent._scene_allows_action(state)
        self.assertFalse(allowed)
        self.assertEqual(reason, "bright_scene_no_keywords")

    def test_scene_allows_action_dark_frame_without_keywords(self):
        agent = PolicyAgent()
        agent.game_keywords = {"poe"}
        state = {
            "timestamp": time.time(),
            "mean": 0.3,
            "text": ["some window"],
        }
        allowed, reason = agent._scene_allows_action(state)
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_scene_blocks_forbidden_cooldown(self):
        agent = PolicyAgent()
        agent.forbidden_until = time.time() + 10
        allowed, reason = agent._scene_allows_action({"timestamp": time.time(), "text": []})
        self.assertFalse(allowed)
        self.assertEqual(reason, "forbidden_cooldown")

    def test_scene_detects_spaced_forbidden_text(self):
        agent = PolicyAgent()
        agent.forbidden_texts = {"shop"}
        state = {
            "timestamp": time.time(),
            "text": ["S H O P"],
        }
        allowed, reason = agent._scene_allows_action(state)
        self.assertFalse(allowed)
        self.assertEqual(reason, "forbidden_ui")

    def test_blend_suppresses_click_when_forbidden(self):
        agent = PolicyAgent()
        agent.latest_state = {"text": ["SHOP"], "timestamp": time.time()}
        agent.forbidden_texts = {"shop"}
        policy_action = {"label": "click_primary"}
        result = agent._blend_with_teacher(policy_action)
        self.assertEqual(result["label"], "mouse_move")

    def test_scene_blocks_when_forbidden_ui_present(self):
        agent = PolicyAgent()
        agent.forbidden_texts = {"shop"}
        state = {
            "timestamp": time.time(),
            "text": ["SHOP"],
        }
        allowed, reason = agent._scene_allows_action(state)
        self.assertFalse(allowed)
        self.assertEqual(reason, "forbidden_ui")

    def test_is_forbidden_target_checks_tags_and_text(self):
        agent = PolicyAgent()
        agent.forbidden_tags = {"shop_button"}
        agent.forbidden_texts = {"shop"}
        self.assertTrue(agent._is_forbidden_target({"tags": ["SHOP_BUTTON"]}))
        self.assertTrue(agent._is_forbidden_target({"text": "Open the shop"}))
        self.assertFalse(agent._is_forbidden_target({"text": "inventory"}))

    def test_teacher_action_blocked_forbidden_text(self):
        agent = PolicyAgent()
        agent.forbidden_texts = {"shop"}
        agent.teacher_action = {"text": "Click the shop"}
        self.assertIsNone(agent._teacher_to_action())
        agent.teacher_action = {"text": "Press W"}
        action = agent._teacher_to_action()
        self.assertEqual(action, {"label": "key_press", "key": "w"})

    def test_auto_respawn_injects_tasks(self):
        agent = PolicyAgent()
        agent.respawn_keywords = {"resurrect"}
        state = {"text": ["Resurrect at checkpoint"], "timestamp": time.time()}
        agent.task_queue.clear()
        agent.current_task = None
        injected = agent._maybe_inject_respawn(state)
        self.assertTrue(injected)
        self.assertTrue(agent.respawn_pending)
        self.assertEqual(len(agent.task_queue), 2)
        self.assertEqual(agent.current_task.get("action_type"), "MOVE_TO")

    def test_auto_respawn_fuzzy_detection(self):
        agent = PolicyAgent()
        agent.respawn_keywords = {"resurrect at checkpoint"}
        texts = ["RESJRICT AT CHECKPOINT"]
        self.assertTrue(agent._scene_has_respawn_text(texts))

    def test_shop_suppress_only_when_death(self):
        agent = PolicyAgent()
        agent.shop_suppress_enabled = True
        agent.shop_suppress_death_only = True
        agent.latest_state = {"text": ["SHOP"], "flags": {}}
        agent.forbidden_texts = {"shop"}
        self.assertFalse(agent._should_suppress_shop())
        agent.latest_state["flags"]["death"] = True
        self.assertTrue(agent._should_suppress_shop())


if __name__ == "__main__":
    unittest.main()
