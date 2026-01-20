#!/usr/bin/env python3
"""Serialize LLM requests via a simple HTTP queue."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

import requests

HOST = os.getenv("LLM_QUEUE_HOST", "0.0.0.0")
PORT = int(os.getenv("LLM_QUEUE_PORT", "9010"))
UPSTREAM_CHAT = os.getenv("LLM_QUEUE_UPSTREAM", "http://127.0.0.1:11434/v1/chat/completions")
UPSTREAM_TIMEOUT = float(os.getenv("LLM_QUEUE_UPSTREAM_TIMEOUT", "120"))
QUEUE_MAX = int(os.getenv("LLM_QUEUE_MAX", "20"))
QUEUE_WAIT_S = float(os.getenv("LLM_QUEUE_WAIT_S", "300"))
LOG_EVERY = float(os.getenv("LLM_QUEUE_LOG_SEC", "10"))

_queue: "queue.Queue[Task]" = queue.Queue(maxsize=max(1, QUEUE_MAX))
_last_log = 0.0


def _models_endpoint() -> str:
    if "/v1/chat/completions" in UPSTREAM_CHAT:
        return UPSTREAM_CHAT.replace("/v1/chat/completions", "/v1/models")
    if UPSTREAM_CHAT.endswith("/v1"):
        return f"{UPSTREAM_CHAT}/models"
    return f"{UPSTREAM_CHAT.rstrip('/')}/models"


def _filter_headers(headers: Dict[str, str]) -> Dict[str, str]:
    keep = {}
    for key in ("authorization", "content-type", "accept"):
        if key in headers:
            keep[key] = headers[key]
    return keep


class Task:
    def __init__(self, path: str, method: str, body: Optional[bytes], headers: Dict[str, str]) -> None:
        self.path = path
        self.method = method
        self.body = body
        self.headers = headers
        self.event = threading.Event()
        self.response: Optional[Tuple[int, Dict[str, str], bytes]] = None
        self.error: Optional[str] = None


def _worker_loop() -> None:
    global _last_log
    while True:
        task = _queue.get()
        if task is None:
            continue
        started = time.time()
        try:
            if task.path.endswith("/v1/models"):
                resp = requests.get(_models_endpoint(), headers=task.headers, timeout=UPSTREAM_TIMEOUT)
            elif task.path.endswith("/v1/chat/completions"):
                resp = requests.post(UPSTREAM_CHAT, data=task.body, headers=task.headers, timeout=UPSTREAM_TIMEOUT)
            else:
                task.response = (404, {"Content-Type": "application/json"}, b'{"error":"not_found"}')
                task.event.set()
                continue
            payload = resp.content or b""
            headers = {"Content-Type": resp.headers.get("Content-Type", "application/json")}
            task.response = (resp.status_code, headers, payload)
        except Exception as exc:
            task.error = str(exc)
            task.response = (502, {"Content-Type": "application/json"}, json.dumps({"error": "upstream_error"}).encode("utf-8"))
        finally:
            task.event.set()
            _queue.task_done()
            now = time.time()
            if now - _last_log >= LOG_EVERY:
                _last_log = now
                qsize = _queue.qsize()
                elapsed = time.time() - started
                print(f"[llm_queue] q={qsize} last={elapsed:.2f}s", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMQueue/0.1"

    def _enqueue_and_wait(self, task: Task) -> Tuple[int, Dict[str, str], bytes]:
        try:
            _queue.put(task, block=False)
        except queue.Full:
            return 429, {"Content-Type": "application/json"}, b'{"error":"queue_full"}'
        waited = task.event.wait(timeout=QUEUE_WAIT_S)
        if not waited or task.response is None:
            return 504, {"Content-Type": "application/json"}, b'{"error":"queue_timeout"}'
        return task.response

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self):  # noqa: N802
        if not self.path.endswith("/v1/models"):
            self.send_response(404)
            self.end_headers()
            return
        headers = _filter_headers({k.lower(): v for k, v in self.headers.items()})
        task = Task(self.path, "GET", None, headers)
        status, resp_headers, payload = self._enqueue_and_wait(task)
        self.send_response(status)
        for key, value in resp_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        if not self.path.endswith("/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        headers = _filter_headers({k.lower(): v for k, v in self.headers.items()})
        body = self._read_body()
        task = Task(self.path, "POST", body, headers)
        status, resp_headers, payload = self._enqueue_and_wait(task)
        self.send_response(status)
        for key, value in resp_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    print(f"[llm_queue] upstream={UPSTREAM_CHAT} host={HOST} port={PORT}", flush=True)
    worker = threading.Thread(target=_worker_loop, daemon=True)
    worker.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
