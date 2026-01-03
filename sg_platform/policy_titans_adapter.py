"""Experimental Titans policy adapter with live memory."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for TitansPolicyAdapter") from exc

try:
    from titans_pytorch import NeuralMemory
except ImportError:  # pragma: no cover
    NeuralMemory = None

logger = logging.getLogger("policy_titans")


class TitansPolicyAdapter:
    """Policy adapter that uses Titans neural memory for action chunks."""

    def __init__(
        self,
        action_space_dim: int,
        device: Optional[str] = None,
        dim: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        if NeuralMemory is None:  # pragma: no cover
            raise RuntimeError("titans-pytorch is required for TitansPolicyAdapter")
        self.action_space_dim = int(action_space_dim)
        self.dim = int(dim or os.getenv("TITANS_DIM", "256"))
        self.chunk_size = int(chunk_size or os.getenv("TITANS_CHUNK", "32"))
        self.device = torch.device(device or os.getenv("TITANS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = self.device.type == "cuda"
        self.allow_projector = os.getenv("TITANS_ALLOW_PROJECTOR", "0") != "0"

        self.memory = NeuralMemory(dim=self.dim, chunk_size=self.chunk_size)
        self.action_head = torch.nn.Linear(self.dim, self.action_space_dim)
        torch.nn.init.zeros_(self.action_head.weight)
        if self.action_head.bias is not None:
            torch.nn.init.zeros_(self.action_head.bias)
        self._mem_state = None
        self._latent_projector: Optional[torch.nn.Module] = None
        self._latent_in_dim: Optional[int] = None
        self.update_interval = max(1, int(os.getenv("TITANS_UPDATE_INTERVAL", "1")))
        self._latent_buffer = []
        self._last_features: Optional[torch.Tensor] = None
        self._lock = threading.Lock()

        self._move_to_device()
        self._maybe_load_memory()

    def predict_chunk(self, observation: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        latent = _extract_latent(observation, state)
        if latent is None:
            logger.warning("TitansPolicyAdapter: missing latent input")
            return _fallback_chunk(self.action_space_dim)

        latent_tensor = self._prepare_latent(latent)
        if latent_tensor is None:
            return _fallback_chunk(self.action_space_dim)
        tokens = latent_tensor.unsqueeze(1)

        with self._lock:
            if self.update_interval > 1:
                self._latent_buffer.append(latent_tensor)
                if len(self._latent_buffer) < self.update_interval:
                    features = self._last_features
                    if features is None:
                        return _fallback_chunk(self.action_space_dim)
                    inference_guard = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
                    with inference_guard():
                        action_vector = self._action_vector_from_features(features)
                    return {
                        "actions": [{"vector": action_vector}],
                        "horizon": 1,
                        "meta": {
                            "backend": "titans",
                            "action_format": "vector",
                            "dim": self.dim,
                            "chunk_size": self.chunk_size,
                            "device": str(self.device),
                            "fp16": self.use_fp16,
                            "titans_version": _get_titans_version(),
                            "update_interval": self.update_interval,
                        },
                    }
                tokens = torch.cat(self._latent_buffer, dim=0).unsqueeze(0)
                self._latent_buffer.clear()

            inference_guard = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with inference_guard():
                try:
                    output = self.memory(tokens, state=self._mem_state)
                except TypeError:
                    output = self.memory(tokens)

                retrieved, new_state = _split_memory_output(output)
                if new_state is not None:
                    self._mem_state = new_state

                features = _select_features(retrieved)
                if features is None:
                    return _fallback_chunk(self.action_space_dim)

                self._last_features = features
                action_vector = self._action_vector_from_features(features)
        return {
            "actions": [{"vector": action_vector}],
            "horizon": 1,
            "meta": {
                "backend": "titans",
                "action_format": "vector",
                "dim": self.dim,
                "chunk_size": self.chunk_size,
                "device": str(self.device),
                "fp16": self.use_fp16,
                "titans_version": _get_titans_version(),
                "update_interval": self.update_interval,
            },
        }

    def save_memory(self) -> Optional[Path]:
        path_str = os.getenv("TITANS_MEM_PATH", "titans_memory.pth")
        if not path_str:
            return None
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mem_state": self._mem_state,
            "dim": self.dim,
            "chunk_size": self.chunk_size,
            "titans_version": _get_titans_version(),
        }
        if self.allow_projector and self._latent_projector is not None:
            payload["projector_state"] = self._latent_projector.state_dict()
            payload["projector_in_dim"] = self._latent_in_dim
        torch.save(payload, path)
        logger.info("Saved Titans memory to %s", path)
        return path

    def _move_to_device(self) -> None:
        self.memory = _move_module(self.memory, self.device)
        self.action_head = _move_module(self.action_head, self.device)
        if self.use_fp16:
            self.action_head = _half_module(self.action_head)
        self.memory.eval()
        self.action_head.eval()
        _freeze_module(self.memory)
        _freeze_module(self.action_head)

    def _maybe_load_memory(self) -> None:
        load_flag = os.getenv("TITANS_LOAD_MEMORY", "0") != "0"
        path_str = os.getenv("TITANS_MEM_PATH", "titans_memory.pth")
        if not load_flag:
            return
        path = Path(path_str)
        if not path.exists():
            logger.warning("Titans memory file not found: %s", path)
            return
        try:
            payload = torch.load(path, map_location=self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load Titans memory: %s", exc)
            return
        if isinstance(payload, dict) and "mem_state" in payload:
            payload_dim = payload.get("dim")
            payload_chunk = payload.get("chunk_size")
            if payload_dim and int(payload_dim) != self.dim:
                logger.warning("Titans memory dim mismatch: %s != %s", payload_dim, self.dim)
                return
            if payload_chunk and int(payload_chunk) != self.chunk_size:
                logger.warning("Titans memory chunk mismatch: %s != %s", payload_chunk, self.chunk_size)
                return
            self._mem_state = payload["mem_state"]
            if self.allow_projector and payload.get("projector_state") and payload.get("projector_in_dim"):
                projector_in_dim = int(payload.get("projector_in_dim"))
                self._latent_in_dim = projector_in_dim
                self._latent_projector = torch.nn.Linear(projector_in_dim, self.dim).to(self.device)
                self._latent_projector.load_state_dict(payload["projector_state"])
                self._latent_projector.eval()
                _freeze_module(self._latent_projector)
        else:
            self._mem_state = payload
        logger.info("Loaded Titans memory from %s", path)

    def _prepare_latent(self, latent: Any) -> Optional[torch.Tensor]:
        tensor = _to_tensor(latent)
        if tensor is None:
            return None
        vector = tensor.reshape(-1)
        if vector.numel() == 0:
            return None
        vector = vector.to(self.device)
        if vector.numel() != self.dim:
            if not self.allow_projector:
                logger.warning(
                    "TitansPolicyAdapter: latent dim %s != %s (set TITANS_ALLOW_PROJECTOR=1 to enable projection)",
                    vector.numel(),
                    self.dim,
                )
                return None
            vector = self._project_latent(vector)
        return vector.unsqueeze(0)

    def _project_latent(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.numel() == self.dim:
            return vector
        if self._latent_projector is None or self._latent_in_dim != int(vector.numel()):
            self._latent_in_dim = int(vector.numel())
            self._latent_projector = torch.nn.Linear(self._latent_in_dim, self.dim).to(self.device)
            self._latent_projector.eval()
            _freeze_module(self._latent_projector)
        proj_input = vector
        try:
            proj_dtype = next(self._latent_projector.parameters()).dtype
            if proj_input.dtype != proj_dtype:
                proj_input = proj_input.to(dtype=proj_dtype)
        except StopIteration:
            pass
        projected = self._latent_projector(proj_input.unsqueeze(0))
        projected = projected.squeeze(0)
        if projected.dtype != torch.float32:
            projected = projected.float()
        return projected

    def _action_vector_from_features(self, features: torch.Tensor) -> list[float]:
        if self.use_fp16:
            features = features.half()
        else:
            features = features.float()
        logits = self.action_head(features)
        return logits.squeeze(0).float().cpu().tolist()


def _extract_latent(observation: Dict[str, Any], state: Dict[str, Any]) -> Optional[Any]:
    if isinstance(state, dict):
        for key in ("latent_state", "latent"):
            if key in state:
                return state.get(key)
    if isinstance(observation, dict):
        return observation.get("latent")
    return None


def _to_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except Exception:  # noqa: BLE001
        return None


def _move_module(module: Any, device: torch.device) -> Any:
    if hasattr(module, "to"):
        return module.to(device)
    return module


def _half_module(module: Any) -> Any:
    if hasattr(module, "half"):
        return module.half()
    return module


def _get_titans_version() -> str:
    try:
        import titans_pytorch

        version = getattr(titans_pytorch, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        import importlib.metadata as metadata

        return metadata.version("titans-pytorch")
    except Exception:
        return "unknown"


def _freeze_module(module: Any) -> None:
    if hasattr(module, "parameters"):
        for param in module.parameters():
            param.requires_grad_(False)


def _split_memory_output(output: Any) -> Tuple[Optional[torch.Tensor], Optional[Any]]:
    if isinstance(output, tuple):
        if len(output) >= 2:
            return output[0], output[1]
        if len(output) == 1:
            return output[0], None
    if isinstance(output, dict):
        return output.get("retrieved"), output.get("state")
    if isinstance(output, torch.Tensor):
        return output, None
    return None, None


def _select_features(retrieved: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if retrieved is None:
        return None
    if retrieved.ndim == 3:
        return retrieved[:, -1, :]
    if retrieved.ndim == 2:
        return retrieved
    if retrieved.ndim == 1:
        return retrieved.unsqueeze(0)
    return None


def _fallback_chunk(action_space_dim: int) -> Dict[str, Any]:
    return {
        "actions": [{"vector": [0.0] * int(action_space_dim)}],
        "horizon": 1,
        "meta": {"backend": "titans", "fallback": True},
    }
