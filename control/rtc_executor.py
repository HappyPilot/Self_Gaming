"""Realtime chunk executor with overlap smoothing and gap fill."""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional


DEFAULT_TICK_SEC = float(os.getenv("RTC_TICK_SEC", "0.1"))
DEFAULT_OVERLAP_STEPS = int(os.getenv("RTC_OVERLAP_STEPS", "2"))
DEFAULT_READY_WINDOW = int(os.getenv("RTC_READY_WINDOW", "50"))
DEFAULT_SAMPLE_EVERY = int(os.getenv("RTC_METRIC_SAMPLE_EVERY", "5"))
DEFAULT_PREFETCH_LEAD = int(os.getenv("RTC_PREFETCH_LEAD_STEPS", "2"))
DEFAULT_GAP_FILL_MODE = os.getenv("RTC_GAP_FILL_MODE", "zero_move")


MetricEmitter = Callable[[str, float, Optional[bool], Optional[dict]], None]


@dataclass
class ActionChunk:
    actions: List[Dict[str, object]]
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_ts: float = field(default_factory=time.time)


class RTCExecutor:
    """Executes action chunks at a fixed tick rate with overlap smoothing."""

    def __init__(
        self,
        publish_action: Callable[[Dict[str, object]], None],
        *,
        chunk_provider: Optional[Callable[[], Optional[List[Dict[str, object]]]]] = None,
        tick_sec: float = DEFAULT_TICK_SEC,
        overlap_steps: int = DEFAULT_OVERLAP_STEPS,
        ready_window: int = DEFAULT_READY_WINDOW,
        sample_every: int = DEFAULT_SAMPLE_EVERY,
        prefetch_lead: int = DEFAULT_PREFETCH_LEAD,
        gap_fill_mode: str = DEFAULT_GAP_FILL_MODE,
        metric_emitter: Optional[MetricEmitter] = None,
    ) -> None:
        self.publish_action = publish_action
        self.chunk_provider = chunk_provider
        self.tick_sec = max(0.01, float(tick_sec))
        self.overlap_steps = max(0, int(overlap_steps))
        self.prefetch_lead = max(0, int(prefetch_lead))
        self.gap_fill_mode = gap_fill_mode
        self.metric_emitter = metric_emitter
        self.sample_every = max(1, int(sample_every))

        self._current: Optional[ActionChunk] = None
        self._current_idx = 0
        self._chunk_queue: Deque[ActionChunk] = deque(maxlen=2)
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._ready_window = deque(maxlen=max(1, int(ready_window)))
        self._tick_count = 0

        self._prefetch_event = threading.Event()
        self._prefetch_thread = None
        if self.chunk_provider is not None:
            self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._prefetch_thread.start()

    def enqueue_chunk(self, actions: List[Dict[str, object]] | ActionChunk) -> None:
        chunk = actions if isinstance(actions, ActionChunk) else ActionChunk(list(actions))
        with self._lock:
            if self._current is None:
                self._current = chunk
                self._current_idx = 0
            else:
                if len(self._chunk_queue) == self._chunk_queue.maxlen:
                    self._chunk_queue.popleft()
                self._chunk_queue.append(chunk)

    def stop(self) -> None:
        self._stop.set()
        self._prefetch_event.set()

    def run(self) -> None:
        while not self._stop.is_set():
            start = time.time()
            self.tick()
            elapsed = time.time() - start
            sleep_for = max(0.0, self.tick_sec - elapsed)
            if sleep_for:
                time.sleep(sleep_for)

    def tick(self) -> Optional[Dict[str, object]]:
        self._maybe_request_next()
        action = self._next_action()
        if action is not None:
            self.publish_action(action)
        self._record_ready_ratio()
        return action

    # Internal -------------------------------------------------------------
    def _prefetch_loop(self) -> None:
        while not self._stop.is_set():
            self._prefetch_event.wait(timeout=0.1)
            if self._stop.is_set():
                return
            if not self._prefetch_event.is_set():
                continue
            self._prefetch_event.clear()
            if self.chunk_provider is None:
                continue
            with self._lock:
                queue_size = len(self._chunk_queue)
                has_current = self._current is not None
            if queue_size > 0:
                continue
            if not has_current:
                pass
            try:
                actions = self.chunk_provider()
            except Exception:
                continue
            if actions:
                self.enqueue_chunk(actions)

    def _maybe_request_next(self) -> None:
        if self.chunk_provider is None:
            return
        with self._lock:
            remaining = None
            if self._current is not None:
                remaining = len(self._current.actions) - self._current_idx
            queue_empty = len(self._chunk_queue) == 0
        if queue_empty and (remaining is None or remaining <= self.prefetch_lead):
            self._prefetch_event.set()

    def _next_action(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._current is None:
                if self._chunk_queue:
                    self._current = self._chunk_queue.popleft()
                    self._current_idx = 0
                else:
                    return self._gap_fill(None)
            if self._current_idx >= len(self._current.actions):
                last_action = self._last_action()
                if self._chunk_queue:
                    next_chunk = self._chunk_queue.popleft()
                    self._current = self._apply_overlap(last_action, next_chunk)
                    self._current_idx = 0
                else:
                    return self._gap_fill(last_action)
            if self._current is None or not self._current.actions:
                return self._gap_fill(None)
            action = self._current.actions[self._current_idx]
            self._current_idx += 1
            return dict(action)

    def _last_action(self) -> Optional[Dict[str, object]]:
        if not self._current or not self._current.actions:
            return None
        idx = max(0, min(self._current_idx - 1, len(self._current.actions) - 1))
        return self._current.actions[idx]

    def _apply_overlap(self, last_action: Optional[Dict[str, object]], next_chunk: ActionChunk) -> ActionChunk:
        if not next_chunk.actions:
            return next_chunk
        first_action = next_chunk.actions[0]
        jerk = self._compute_jerk(last_action, first_action)
        if jerk is not None:
            self._emit_metric("control/chunk_boundary_jerk", jerk, ok=None, tags=None)
        if self.overlap_steps <= 0:
            return next_chunk
        blended = self._blend_actions(last_action, first_action, self.overlap_steps)
        if not blended:
            return next_chunk
        merged = blended + list(next_chunk.actions)
        return ActionChunk(actions=merged, chunk_id=next_chunk.chunk_id, created_ts=next_chunk.created_ts)

    def _blend_actions(
        self,
        last_action: Optional[Dict[str, object]],
        first_action: Dict[str, object],
        steps: int,
    ) -> List[Dict[str, object]]:
        if steps <= 0:
            return []
        last_vec = self._action_vector(last_action)
        first_vec = self._action_vector(first_action)
        if last_vec is None or first_vec is None:
            return []
        blended: List[Dict[str, object]] = []
        for i in range(steps):
            alpha = (i + 1) / (steps + 1)
            dx = last_vec[0] + (first_vec[0] - last_vec[0]) * alpha
            dy = last_vec[1] + (first_vec[1] - last_vec[1]) * alpha
            payload = dict(first_action)
            payload["dx"] = float(dx)
            payload["dy"] = float(dy)
            blended.append(payload)
        return blended

    def _gap_fill(self, last_action: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        mode = (self.gap_fill_mode or "zero_move").lower()
        if mode == "hold_last" and last_action is not None:
            return dict(last_action)
        if mode == "wait":
            return {"action": "wait"}
        if mode == "noop":
            return None
        return {"action": "mouse_move", "dx": 0, "dy": 0}

    def _action_vector(self, action: Optional[Dict[str, object]]) -> Optional[tuple[float, float]]:
        if not action:
            return None
        dx = action.get("dx")
        dy = action.get("dy")
        if isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
            return float(dx), float(dy)
        return None

    def _compute_jerk(
        self,
        prev_action: Optional[Dict[str, object]],
        next_action: Optional[Dict[str, object]],
    ) -> Optional[float]:
        prev_vec = self._action_vector(prev_action)
        next_vec = self._action_vector(next_action)
        if prev_vec is None or next_vec is None:
            return None
        return ((next_vec[0] - prev_vec[0]) ** 2 + (next_vec[1] - prev_vec[1]) ** 2) ** 0.5

    def _record_ready_ratio(self) -> None:
        with self._lock:
            ready = len(self._chunk_queue) > 0
        self._ready_window.append(1 if ready else 0)
        self._tick_count += 1
        if self._tick_count % self.sample_every != 0:
            return
        if not self._ready_window:
            return
        ratio = sum(self._ready_window) / len(self._ready_window)
        self._emit_metric(
            "control/next_chunk_ready_ratio",
            ratio,
            ok=None,
            tags={"window": len(self._ready_window)},
        )

    def _emit_metric(self, name: str, value: float, ok: Optional[bool], tags: Optional[dict]) -> None:
        if self.metric_emitter is None:
            return
        self.metric_emitter(name, float(value), ok, tags)
