# Death Click Cooldown + Jetson Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a death-click cooldown in `policy_brain` and validate locally + on Jetson, then run a 30-minute monitor.

**Architecture:** Add a timestamp-based cooldown gate in the heuristic policy model for death clicks; keep policy_agent respawn macro unchanged. Validate with unit tests, then sync to Jetson and run monitoring.

**Tech Stack:** Python, pytest, ssh, MQTT agents.

---

### Task 1: Add failing test for death-click cooldown

**Files:**
- Modify: `agents/tests/test_policy_brain_respawn_target.py`

**Step 1: Write the failing test**

```python
def test_death_click_cooldown_returns_wait():
    import time
    from policy_brain import HeuristicPolicyModel

    model = HeuristicPolicyModel()
    obs = {"flags": {"death": True}}

    first = model(obs)
    second = model(obs)  # immediately after first

    assert first["action"] in {"click_primary", "click_secondary"}
    assert second["action"] == "wait"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=agents python3 -m pytest agents/tests/test_policy_brain_respawn_target.py -q`
Expected: FAIL because cooldown not implemented yet.

---

### Task 2: Implement cooldown in policy_brain

**Files:**
- Modify: `agents/policy_brain.py`

**Step 1: Write minimal implementation**

```python
DEATH_CLICK_COOLDOWN = float(os.environ.get("DEATH_CLICK_COOLDOWN", "2.0"))

class HeuristicPolicyModel:
    def __init__(self):
        self.last_death_click_ts = 0.0

    def __call__(self, obs):
        if obs.get("flags", {}).get("death"):
            now = time.time()
            if now - self.last_death_click_ts < DEATH_CLICK_COOLDOWN:
                return {"action": "wait", "reason": "death_cooldown"}
            self.last_death_click_ts = now
            return {"action": "click_primary", "target": {"x": 0.5, "y": 0.82}, "label": "respawn_button"}
```

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=agents python3 -m pytest agents/tests/test_policy_brain_respawn_target.py -q`
Expected: PASS

---

### Task 3: Run local test set

**Files:**
- No file changes

**Step 1: Run tests**

Run: `PYTHONPATH=agents python3 -m pytest agents/tests/test_policy_brain_respawn_target.py agents/tests/test_policy_respawn_macro.py agents/tests/test_policy_wait_task.py agents/tests/test_policy_scene_allows_action.py -q`
Expected: PASS

---

### Task 4: Sync to Jetson and run tests

**Files:**
- Sync modified: `agents/policy_brain.py`, `agents/tests/test_policy_brain_respawn_target.py`

**Step 1: Rsync changes**

Run: `rsync -az --delete agents/policy_brain.py agents/tests/test_policy_brain_respawn_target.py dima@10.0.0.68:~/self-gaming/agents/`

**Step 2: Run tests on Jetson**

Run: `ssh dima@10.0.0.68 'cd ~/self-gaming && PYTHONPATH=agents python3 -m pytest agents/tests/test_policy_brain_respawn_target.py agents/tests/test_policy_respawn_macro.py agents/tests/test_policy_wait_task.py agents/tests/test_policy_scene_allows_action.py -q'`
Expected: PASS

---

### Task 5: Start 30-minute monitor on Jetson

**Files:**
- No code changes

**Step 1: Run monitor**

Run: `ssh dima@10.0.0.68 'cd ~/self-gaming && ACTION_WINDOW_SEC=60 SAMPLE_INTERVAL_SEC=180 SAMPLE_COUNT=10 python3 agents/monitor_game_actions.py'`
Expected: Monitor starts and runs for ~30 minutes.
