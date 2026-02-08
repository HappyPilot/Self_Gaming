# Game-Agnostic Combat + Vision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace naive combat targeting with game-aware strategy, key semantics, skill profiling, and enemy learning.

**Architecture:** Extend onboarding to produce a richer control profile and UI layout; add skill profiler; update scene/policy to consume per-game enemy labels and key groups; keep actions constrained by profile.

**Tech Stack:** Python, MQTT, existing agents (game_onboarding_agent, scene_agent, policy_agent), llama.cpp (text-only).

---

### Task 1: Add Control Profile v2 schema helpers

**Files:**
- Modify: `agents/control_profile.py`
- Create: `agents/utils/control_profile_v2.py`
- Test: `tests/test_control_profile_v2.py`

**Step 1: Write the failing test**

```python
# tests/test_control_profile_v2.py
import unittest
from agents.utils.control_profile_v2 import normalize_profile_v2

class ControlProfileV2Test(unittest.TestCase):
    def test_normalize_profile_groups(self):
        profile = {
            "game_id": "test_game",
            "bindings": [
                {"action": "inventory_open", "keys": ["i"], "category": "ui"},
                {"action": "skill_1", "keys": ["q"], "category": "combat"},
                {"action": "special_combo", "keys": ["shift+1"], "category": "combat"},
            ],
        }
        normalized = normalize_profile_v2(profile)
        self.assertIn("allowed_keys_gameplay", normalized)
        self.assertIn("blocked_keys_aux", normalized)
        self.assertIn("allowed_combos_gameplay", normalized)
        self.assertIn("i", normalized["blocked_keys_aux"])
        self.assertIn("q", normalized["allowed_keys_gameplay"])
        self.assertIn("shift+1", normalized["allowed_combos_gameplay"])
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_control_profile_v2.py`  
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
# agents/utils/control_profile_v2.py
from __future__ import annotations

AUX_CATEGORIES = {"ui", "social", "system"}
GAMEPLAY_CATEGORIES = {"movement", "combat", "interaction"}

def _split_keys_and_combos(values: list[str]) -> tuple[list[str], list[str]]:
    keys = []
    combos = []
    for value in values:
        if "+" in value:
            combos.append(value)
        else:
            keys.append(value)
    return keys, combos

def normalize_profile_v2(profile: dict) -> dict:
    profile = dict(profile or {})
    bindings = profile.get("bindings") or []
    allowed = set()
    blocked = set()
    allowed_combos = set()
    blocked_combos = set()
    for binding in bindings:
        category = str(binding.get("category") or "").lower()
        keys = [str(k).lower() for k in (binding.get("keys") or []) if k]
        plain_keys, combo_keys = _split_keys_and_combos(keys)
        if category in AUX_CATEGORIES:
            blocked.update(plain_keys)
            blocked_combos.update(combo_keys)
        elif category in GAMEPLAY_CATEGORIES:
            allowed.update(plain_keys)
            allowed_combos.update(combo_keys)
    profile["allowed_keys_gameplay"] = sorted(allowed)
    profile["blocked_keys_aux"] = sorted(blocked)
    profile["allowed_combos_gameplay"] = sorted(allowed_combos)
    profile["blocked_combos_aux"] = sorted(blocked_combos)
    return profile

def apply_profile_v2(profile: dict) -> dict:
    profile = normalize_profile_v2(profile)
    # default allowed_keys to gameplay keys if not explicitly set
    if not profile.get("allowed_keys"):
        profile["allowed_keys"] = list(profile.get("allowed_keys_gameplay") or [])
    return profile
```

Update `agents/control_profile.py` to call `apply_profile_v2` inside `safe_profile()` and
`upsert_profile()` so stored profiles include gameplay-only allowed keys.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_control_profile_v2.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add agents/utils/control_profile_v2.py tests/test_control_profile_v2.py agents/control_profile.py
git commit -m "feat: add control profile v2 normalization"
```

---

### Task 2: Extend onboarding to emit Control Profile v2 + key meanings

**Files:**
- Modify: `agents/llm_client.py`
- Modify: `agents/game_onboarding_agent.py`
- Test: `tests/test_onboarding_profile_v2.py`

**Step 1: Write the failing test**

```python
# tests/test_onboarding_profile_v2.py
import unittest
from agents.game_onboarding_agent import GameOnboardingAgent

class OnboardingProfileV2Test(unittest.TestCase):
    def test_profile_to_controls_uses_gameplay_keys_only(self):
        agent = GameOnboardingAgent()
        profile = {
            "allow_mouse_move": True,
            "allow_primary": True,
            "bindings": [
                {"action": "inventory_open", "keys": ["i"], "category": "ui"},
                {"action": "skill_1", "keys": ["q"], "category": "combat"},
            ],
        }
        controls = agent._profile_to_controls(profile)
        self.assertIn("key_q", controls)
        self.assertNotIn("key_i", controls)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_onboarding_profile_v2.py`  
Expected: FAIL (key_q missing because profile bindings are ignored).

**Step 3: Update LLM prompt and onboarding**

- Update `fetch_control_profile` to request `bindings[]` with `action, keys, category, purpose, contexts, risk, confidence`.
- Save bindings into `profile` and pass through `normalize_profile_v2`.
- Use gameplay-only keys when building controls.
- Include `allowed_keys_gameplay` and combo lists in `game/schema.profile`.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_onboarding_profile_v2.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add agents/llm_client.py agents/game_onboarding_agent.py tests/test_onboarding_profile_v2.py
git commit -m "feat: emit control profile v2 from onboarding"
```

---

### Task 3: Skill profiler agent (OCR + LLM)

**Files:**
- Create: `agents/skill_profiler_agent.py`
- Modify: `docker-compose.yml` (add service)
- Test: `tests/test_skill_profiler_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_skill_profiler_agent.py
import unittest
from agents.skill_profiler_agent import classify_skill_type

class SkillProfilerTest(unittest.TestCase):
    def test_classify_skill_type(self):
        result = classify_skill_type("Deals damage in an area around you")
        self.assertEqual(result, "aoe")
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_profiler_agent.py`  
Expected: FAIL (module missing).

**Step 3: Implement minimal classifier + LLM hook**

- Implement a lightweight heuristic classifier (fallback).
- Add LLM request path for richer classification.
- Publish results to `mem/store` and `game/schema`.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_skill_profiler_agent.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add agents/skill_profiler_agent.py tests/test_skill_profiler_agent.py docker-compose.yml
git commit -m "feat: add skill profiler agent"
```

---

### Task 4: Enemy label learning in scene_agent

**Files:**
- Modify: `agents/scene_agent.py`
- Test: `tests/test_enemy_label_learning.py`

**Step 1: Write failing test**

```python
# tests/test_enemy_label_learning.py
import unittest
from agents.scene_agent import _update_dynamic_enemy_labels

class EnemyLabelLearningTest(unittest.TestCase):
    def test_update_dynamic_enemy_labels(self):
        labels = set(["enemy"])
        objects = [{"label": "goblin", "bbox": [0.1,0.1,0.2,0.2]}]
        bars = [{"bbox": [0.1,0.1,0.2,0.2]}]
        updated = _update_dynamic_enemy_labels(labels, objects, bars)
        self.assertIn("goblin", updated)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_enemy_label_learning.py`  
Expected: FAIL (function missing).

**Step 3: Implement label overlap learning**

- Add helper to count overlaps between object boxes and enemy_bars.
- Promote labels with sufficient overlap frequency.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_enemy_label_learning.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add agents/scene_agent.py tests/test_enemy_label_learning.py
git commit -m "feat: learn enemy labels via bar overlap"
```

---

### Task 5: Combat strategy in policy_agent

**Files:**
- Modify: `agents/policy_agent.py`
- Create: `agents/utils/combat_targeting.py`
- Test: `tests/test_combat_targeting.py`

**Step 1: Write failing test**

```python
# tests/test_combat_targeting.py
import unittest
from agents.utils.combat_targeting import pick_enemy_target

class CombatTargetingTest(unittest.TestCase):
    def test_cluster_targeting(self):
        enemies = [
            {"bbox": [0.4,0.4,0.45,0.45]},
            {"bbox": [0.5,0.4,0.55,0.45]},
        ]
        target = pick_enemy_target(enemies, player_center=(0.5, 0.5))
        self.assertIsNotNone(target)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_combat_targeting.py`  
Expected: FAIL if signature differs.

**Step 3: Implement combat strategy state machine**

- Move enemy targeting logic into `agents/utils/combat_targeting.py`.
- Add combat states and transitions in policy_agent.
- Use skill type info to choose AOE vs single.
- For boss: keep distance + circle movement.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_combat_targeting.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add agents/policy_agent.py agents/utils/combat_targeting.py tests/test_combat_targeting.py
git commit -m "feat: add combat strategy state machine"
```

---

### Task 6: Documentation + wiring

**Files:**
- Modify: `docs/overview/system_map.md`
- Modify: `docs/overview/howto.md`
- Add: `docs/overview/jetson.md`
- Modify: `docs/mqtt/topics.md`

**Step 1: Update docs**

Add:
- new topics for skill_profiler
- control profile v2 details
- enemy learning notes
 - Jetson environment notes (if updated)

**Step 2: Commit**

```bash
git add docs/overview/system_map.md docs/overview/howto.md docs/overview/jetson.md docs/mqtt/topics.md
git commit -m "docs: update combat/vision design details"
```

---

Plan complete.
