#!/usr/bin/env python3
"""Lightweight MQTT RPC helper for mem_agent queries."""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt


class MemRPC:
    """Utility class that keeps a lightweight MQTT client for mem_agent RPC."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        query_topic: str = "mem/query",
        reply_topic: str = "mem/response",
        client_id: Optional[str] = None,
    ) -> None:
        self._client = mqtt.Client(client_id=client_id or f"mem_rpc_{uuid.uuid4().hex[:8]}")
        self._client.on_message = self._on_message
        self._client.connect(host, port, keepalive=30)
        self._client.subscribe(reply_topic)
        self._query_topic = query_topic
        self._reply_topic = reply_topic
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._client.loop_start()

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    # ------------------------------------------------------------------ RPC
    def _on_message(self, _client, _userdata, msg):
        payload = msg.payload.decode("utf-8", "ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload}
        request_id = data.get("request_id")
        if not request_id:
            return
        with self._lock:
            pending = self._pending.get(request_id)
        if pending:
            pending["response"] = data
            pending["event"].set()

    def query(self, payload: Dict[str, Any], timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Publish a query and wait for the mem_agent response."""

        request_id = payload.get("request_id") or f"memreq_{uuid.uuid4().hex[:8]}"
        payload["request_id"] = request_id
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event}
        self._client.publish(self._query_topic, json.dumps(payload))
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            return None
        with self._lock:
            entry = self._pending.pop(request_id, None)
        if not entry:
            return None
        return entry.get("response")
