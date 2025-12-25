"""VLA training stub (imitation baseline + action chunking head)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for training/vla/train_stub.py") from exc

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from training.vla.model import ActionChunkMLP
from training.vla.utils import encode_action, iter_jsonl
from world_state.encoder import FrameEncoder

logging.basicConfig(level=os.getenv("VLA_TRAIN_LOG_LEVEL", "INFO"))
logger = logging.getLogger("vla_train_stub")

DEFAULT_ACTION_DIM = 4


def _build_action_chunk(actions: List[Dict[str, Any]], chunk_size: int, action_dim: int) -> List[float]:
    if not actions:
        return [0.0] * (chunk_size * action_dim)
    if action_dim != DEFAULT_ACTION_DIM:
        logger.warning("action_dim=%s is custom; only dx/dy/click/key are encoded", action_dim)
    ordered = sorted(actions, key=lambda item: item.get("timestamp", 0.0))
    chunk: List[float] = []
    for idx in range(chunk_size):
        if idx < len(ordered):
            entry = ordered[idx]
            payload = entry.get("action") if isinstance(entry.get("action"), dict) else entry
            if not isinstance(payload, dict):
                payload = {}
            chunk.extend(encode_action(payload, action_dim))
        else:
            chunk.extend([0.0] * action_dim)
    return chunk


def _load_samples(dataset_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in iter_jsonl(dataset_path):
        samples.append(row)
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def _build_tensors(
    samples: List[Dict[str, Any]],
    encoder: FrameEncoder,
    chunk_size: int,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: List[np.ndarray] = []
    targets: List[List[float]] = []
    skipped = 0
    for sample in samples:
        frame_path = sample.get("frame_path")
        if not frame_path:
            skipped += 1
            continue
        path = Path(frame_path)
        if not path.exists():
            skipped += 1
            continue
        data = path.read_bytes()
        latent = encoder.encode_bytes(data)
        if latent is None:
            skipped += 1
            continue
        actions = sample.get("actions")
        if not isinstance(actions, list):
            actions = []
        target = _build_action_chunk(actions, chunk_size, action_dim)
        features.append(latent.astype(np.float32))
        targets.append(target)
    if skipped:
        logger.info("Skipped %s samples (missing frames/latents)", skipped)
    if not features:
        raise RuntimeError("No usable samples (check frames or OpenCV availability)")
    x = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
    y = torch.tensor(np.array(targets, dtype=np.float32), dtype=torch.float32)
    return x, y


def _configure_mlflow() -> Optional[Any]:
    if mlflow is None:
        return None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "self_gaming")
    run_name = os.getenv("MLFLOW_RUN_NAME", "vla_stub")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def _log_param(key: str, value: Any) -> None:
    if mlflow is None:
        return
    mlflow.log_param(key, value)


def _log_metric(key: str, value: float, step: int) -> None:
    if mlflow is None:
        return
    mlflow.log_metric(key, value, step=step)


def _log_artifact(path: Path) -> None:
    if mlflow is None:
        return
    mlflow.log_artifact(str(path))


def _export_onnx(model: torch.nn.Module, output_path: Path, input_dim: int) -> None:
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["latent"],
        output_names=["action_chunk"],
        opset_version=13,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stub VLA policy")
    parser.add_argument("--dataset", required=True, help="Path to prepared JSONL dataset")
    parser.add_argument("--output-dir", default="training/vla/output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = _configure_mlflow()
    try:
        _log_param("epochs", args.epochs)
        _log_param("batch_size", args.batch_size)
        _log_param("lr", args.lr)
        _log_param("chunk_size", args.chunk_size)
        _log_param("action_dim", args.action_dim)
        _log_param("latent_dim", args.latent_dim)
        _log_param("frame_size", args.frame_size)
        _log_param("hidden_dim", args.hidden_dim)
        _log_param("depth", args.depth)
        _log_param("dropout", args.dropout)

        encoder = FrameEncoder(frame_size=args.frame_size, latent_dim=args.latent_dim)
        samples = _load_samples(dataset_path, args.max_samples)
        if not samples:
            raise RuntimeError("Dataset is empty")

        x, y = _build_tensors(samples, encoder, args.chunk_size, args.action_dim)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = ActionChunkMLP(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
        )
        device = torch.device(args.device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(loader))
            logger.info("Epoch %s/%s loss=%.6f", epoch, args.epochs, avg_loss)
            _log_metric("train_loss", avg_loss, step=epoch)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": vars(args),
            "created_at": time.time(),
        }
        ckpt_path = output_dir / f"vla_stub_{int(time.time())}.pt"
        torch.save(checkpoint, ckpt_path)
        _log_artifact(ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)

        if args.export_onnx or args.onnx_path:
            onnx_path = Path(args.onnx_path) if args.onnx_path else output_dir / "vla_stub.onnx"
            model.eval()
            _export_onnx(model, onnx_path, input_dim=x.shape[1])
            _log_artifact(onnx_path)
            logger.info("Exported ONNX: %s", onnx_path)
    finally:
        if run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
