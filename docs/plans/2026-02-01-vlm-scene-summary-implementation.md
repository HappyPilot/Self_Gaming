# VLM Scene Summary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a VLM-driven scene summary pipeline (llama.cpp on Mac) and integrate it into teacher_agent decisions, with 30‑minute behavior logs including VLM context.

**Architecture:** Introduce a new `vlm_summary_agent` that subscribes to frames, calls the VLM endpoint, and publishes structured summaries to `scene/summary`. Teacher agent subscribes to these summaries and injects them into the LLM prompt as context hints while preserving safety rules.

**Tech Stack:** Python, paho-mqtt, llama.cpp OpenAI-compatible endpoint, pytest.

---

### Task 1: Define VLM summary schema + tests

**Files:**
- Create: `agents/tests/test_vlm_summary_agent.py`
- Modify: `schemas/` (optional if schema json is added)

**Step 1: Write the failing test**
```python
from agents import vlm_summary_agent as vsa


def test_parse_vlm_summary_payload():
    payload = {
        "game": "path_of_exile",
        "summary": "town hub, NPCs nearby",
        "player_state": "healthy",
        "enemies": "none",
        "objectives": "find stash",
        "ui": "inventory closed",
        "risk": "low",
        "recommended_intent": "move to stash",
        "timestamp": 123.0,
    }
    parsed = vsa.normalize_summary(payload)
    assert parsed["game"] == "path_of_exile"
    assert parsed["risk"] == "low"
    assert "summary" in parsed
```

**Step 2: Run test to verify it fails**
Run: `pytest agents/tests/test_vlm_summary_agent.py::test_parse_vlm_summary_payload -v`
Expected: FAIL (module not found / function missing).

---

### Task 2: Implement vlm_summary_agent

**Files:**
- Create: `agents/vlm_summary_agent.py`

**Step 1: Write minimal implementation**
```python
# agent subscribes to vision/frame/preview, calls VLM, publishes to scene/summary
# includes normalize_summary() + safe JSON parsing
```

**Step 2: Run test to verify it passes**
Run: `pytest agents/tests/test_vlm_summary_agent.py::test_parse_vlm_summary_payload -v`
Expected: PASS.

**Step 3: Commit**
```bash
git add agents/vlm_summary_agent.py agents/tests/test_vlm_summary_agent.py
git commit -m "Add VLM summary agent scaffold"
```

---

### Task 3: Wire teacher_agent to scene/summary

**Files:**
- Modify: `agents/teacher_agent.py`

**Step 1: Write failing test**
```python
from agents import teacher_agent as ta


def test_teacher_includes_vlm_summary_in_scene():
    agent = ta.TeacherAgent()
    agent.vlm_summary = {"summary": "enemy boss", "risk": "high"}
    scene = {"ok": True, "text": ["dummy"], "enemies": []}
    summary = agent._build_scene_summary(scene, [], [])
    assert "VLM" in summary
```

**Step 2: Run test to verify it fails**
Run: `pytest agents/tests/test_teacher_agent.py::test_teacher_includes_vlm_summary_in_scene -v`
Expected: FAIL.

**Step 3: Implement**
- Subscribe to `scene/summary` topic.
- Store last summary on `teacher_agent`.
- Inject into `_build_scene_summary`.

**Step 4: Run test to verify it passes**
Run: `pytest agents/tests/test_teacher_agent.py::test_teacher_includes_vlm_summary_in_scene -v`
Expected: PASS.

**Step 5: Commit**
```bash
git add agents/teacher_agent.py agents/tests/test_teacher_agent.py
git commit -m "Use VLM scene summaries in teacher agent"
```

---

### Task 4: Add VLM logging + 30‑minute monitoring update

**Files:**
- Modify: `/tmp/monitor_game_actions.py` (Jetson script only)

**Step 1: Extend monitor**
- Subscribe to `scene/summary` and log latest VLM summary per action.
- Add `summary_staleness_sec` and `vlm_latency_ms` fields.

**Step 2: 30‑minute run on Jetson**
Run: `MONITOR_DURATION_SEC=1800 /tmp/monitor_game_actions.py`
Expected: `summary.json` + `events.jsonl` contains VLM fields.

---

### Task 5: Deploy + validate on Jetson

**Files:**
- None (ops only)

**Step 1: Start vlm_summary_agent**
Run: `python3 agents/vlm_summary_agent.py`

**Step 2: Verify scene/summary topic**
Run: `mosquitto_sub -t scene/summary -C 1 -v`

**Step 3: Run 30‑minute monitor**
Validate metrics and screenshot samples.

