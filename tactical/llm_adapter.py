"""Tactical LLM adapter for strategy updates."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("tactical_llm")

_DEFAULT_ENDPOINT = os.getenv("TACTICAL_LLM_ENDPOINT") or os.getenv(
    "LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"
)
_DEFAULT_MODEL = os.getenv("TACTICAL_LLM_MODEL") or os.getenv("LLM_MODEL", "llama")
_DEFAULT_TIMEOUT = float(os.getenv("TACTICAL_LLM_TIMEOUT_SEC", "8.0"))
_DEFAULT_TEMPERATURE = float(os.getenv("TACTICAL_LLM_TEMPERATURE", "0.2"))
_DEFAULT_BACKEND = os.getenv("TACTICAL_LLM_BACKEND", "remote_http").strip().lower()
_DEFAULT_MODE = os.getenv("TACTICAL_DEFAULT_MODE", "scan")
_DEFAULT_API_KEY = os.getenv("TACTICAL_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))

_SYSTEM_PROMPT = os.getenv(
    "TACTICAL_LLM_SYSTEM_PROMPT",
    "You are a tactical planner. Return JSON only with keys: global_strategy, targets, cooldowns.",
)


class TacticalLLMAdapter:
    """Selects an LLM backend and returns strategy updates."""

    def __init__(self, backend: Optional["TacticalLLMBackend"] = None) -> None:
        self.backend = backend or _build_backend(_DEFAULT_BACKEND)

    def plan(self, summary: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a strategy_update dict for shared state."""
        try:
            update = self.backend.plan(summary=summary, context=context)
        except Exception as exc:
            logger.warning("Tactical LLM plan failed: %s", exc)
            update = None
        return _normalize_update(update)


class TacticalLLMBackend:
    """Interface for tactical backends."""

    def plan(self, summary: Any, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class RemoteHTTPBackend(TacticalLLMBackend):
    """HTTP backend that calls a chat-completions compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str = _DEFAULT_MODEL,
        timeout_sec: float = _DEFAULT_TIMEOUT,
        api_key: str = _DEFAULT_API_KEY,
        temperature: float = _DEFAULT_TEMPERATURE,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_sec = timeout_sec
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt

    def plan(self, summary: Any, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        payload = _build_request(summary, context, self.model, self.temperature, self.system_prompt)
        response = _post_json(self.endpoint, payload, self.timeout_sec, self.api_key)
        return _extract_update(response)


class TrtLLMBackend(RemoteHTTPBackend):
    """TensorRT-LLM backend alias (HTTP endpoint)."""

    def __init__(self) -> None:
        endpoint = os.getenv("TACTICAL_TRT_LLM_ENDPOINT", _DEFAULT_ENDPOINT)
        model = os.getenv("TACTICAL_TRT_LLM_MODEL", _DEFAULT_MODEL)
        timeout_sec = float(os.getenv("TACTICAL_TRT_LLM_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT)))
        api_key = os.getenv("TACTICAL_TRT_LLM_API_KEY", _DEFAULT_API_KEY)
        temperature = float(os.getenv("TACTICAL_TRT_LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
        super().__init__(
            endpoint=endpoint,
            model=model,
            timeout_sec=timeout_sec,
            api_key=api_key,
            temperature=temperature,
            system_prompt=_SYSTEM_PROMPT,
        )


def _build_backend(kind: str) -> TacticalLLMBackend:
    if kind == "remote_http":
        return RemoteHTTPBackend(endpoint=_DEFAULT_ENDPOINT)
    if kind == "trt_llm":
        return TrtLLMBackend()
    raise ValueError(f"Unknown tactical LLM backend: {kind}")


def _build_request(
    summary: Any,
    context: Optional[Dict[str, Any]],
    model: str,
    temperature: float,
    system_prompt: str,
) -> Dict[str, Any]:
    summary_text = _format_payload(summary)
    context_text = _format_payload(context) if context else ""
    user_lines = ["Summary:", summary_text]
    if context_text:
        user_lines.extend(["", "Context:", context_text])
    user_lines.append("")
    user_lines.append("Return a JSON object with keys: global_strategy, targets, cooldowns.")
    user_prompt = "\n".join([line for line in user_lines if line is not None])
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }


def _post_json(endpoint: str, payload: Dict[str, Any], timeout_sec: float, api_key: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if requests is not None:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_sec)
        response.raise_for_status()
        return response.json()
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _extract_update(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict):
        update = data.get("strategy_update")
        if isinstance(update, dict):
            return update
        if _looks_like_update(data):
            return data
    content = _extract_content(data)
    if not content:
        return None
    return _parse_update_from_text(content)


def _extract_content(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    if "choices" in data:
        try:
            choice = data.get("choices", [])[0]
        except IndexError:
            choice = None
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
            if isinstance(choice.get("text"), str):
                return choice["text"].strip()
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    content = data.get("content")
    if isinstance(content, str):
        return content.strip()
    return None


def _parse_update_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            value = json.loads(snippet)
            return value if isinstance(value, dict) else None
        except Exception:
            return None


def _format_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _looks_like_update(payload: Dict[str, Any]) -> bool:
    return any(key in payload for key in ("global_strategy", "targets", "cooldowns"))


def _normalize_update(update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    now = time.time()
    if not isinstance(update, dict):
        return {
            "global_strategy": {"mode": _DEFAULT_MODE, "ts": now},
            "targets": {},
            "cooldowns": {},
        }
    normalized = dict(update)
    global_strategy = update.get("global_strategy")
    if not isinstance(global_strategy, dict):
        global_strategy = {"mode": global_strategy or _DEFAULT_MODE}
    global_strategy.setdefault("mode", _DEFAULT_MODE)
    global_strategy.setdefault("ts", now)
    normalized["global_strategy"] = global_strategy
    if not isinstance(normalized.get("targets"), dict):
        normalized["targets"] = {}
    if not isinstance(normalized.get("cooldowns"), dict):
        normalized["cooldowns"] = {}
    return normalized
