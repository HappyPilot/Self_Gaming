#!/usr/bin/env python3
"""Export a vision encoder to ONNX for TensorRT."""
from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel
except Exception as exc:  # pragma: no cover - manual script
    raise SystemExit(
        "Failed to import transformers. Install first: pip install transformers -- reason: %s" % exc
    )

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("export_vision_encoder_onnx")

MODEL_ID_MAP = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "clip-vit-b32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
}


def _resolve_model_id(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_ID_MAP.get(normalized, name)


def _extract_embedding(outputs, pooling: str) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    elif hasattr(outputs, "image_embeds") and isinstance(outputs.image_embeds, torch.Tensor):
        tensor = outputs.image_embeds
    elif hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
        tensor = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
        tensor = outputs.last_hidden_state
    elif isinstance(outputs, dict):
        tensor = None
        for key in ("image_embeds", "pooler_output", "last_hidden_state", "embeddings", "features", "feat", "output"):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        if tensor is None:
            raise RuntimeError(f"Could not find tensor output in keys: {list(outputs.keys())}")
    elif isinstance(outputs, (tuple, list)):
        tensor = next((item for item in outputs if isinstance(item, torch.Tensor)), None)
        if tensor is None:
            raise RuntimeError("Could not find tensor output in tuple/list output")
    else:
        raise RuntimeError(f"Unsupported output type: {type(outputs)}")

    if tensor.ndim == 3:
        if pooling == "cls":
            return tensor[:, 0, :]
        return tensor.mean(dim=1)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim > 2:
        return tensor.flatten(1)
    return tensor


class EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_size: int, mean: Sequence[float], std: Sequence[float], pooling: str) -> None:
        super().__init__()
        self.model = model
        self.input_size = int(input_size)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.pooling = pooling
        signature = inspect.signature(self.model.forward)
        self._use_pixel_values = "pixel_values" in signature.parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Expected 4D input, got shape {tuple(x.shape)}")
        x = x.to(self.mean.dtype)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self._use_pixel_values:
            outputs = self.model(pixel_values=x)
        else:
            outputs = self.model(x)
        return _extract_embedding(outputs, self.pooling)


def export_onnx(
    model_id: str,
    output: Path,
    input_size: int,
    pooling: str,
    batch_size: int,
    opset: int,
    meta_path: Optional[Path],
) -> Path:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = meta_path or output.with_suffix(".meta.json")

    logger.info("Loading model %s (device=cpu)", model_id)
    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Image processor not found for %s (%s); using default mean/std.", model_id, exc)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(config, "model_type", "") == "clip":
        model = CLIPVisionModel.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406]) if processor else [0.485, 0.456, 0.406]
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225]) if processor else [0.229, 0.224, 0.225]

    wrapper = EncoderWrapper(model, input_size, mean, std, pooling)
    wrapper.eval()

    example = torch.zeros((batch_size, 3, input_size, input_size), dtype=torch.float32)
    with torch.inference_mode():
        output_tensor = wrapper(example)
    embed_dim = int(output_tensor.shape[-1])
    logger.info("Embedding shape: %s (dim=%s)", tuple(output_tensor.shape), embed_dim)

    torch.onnx.export(
        wrapper,
        example,
        output.as_posix(),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
    )

    meta = {
        "model_id": model_id,
        "input_size": input_size,
        "pooling": pooling,
        "embed_dim": embed_dim,
        "dtype": "float32",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved ONNX: %s", output)
    logger.info("Saved metadata: %s", meta_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vision encoder to ONNX")
    parser.add_argument("--model", default="dinov2-small", help="Model alias or HF model id")
    parser.add_argument("--input", type=int, default=224, help="Input size (default: 224)")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Token pooling")
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    parser.add_argument("--meta", help="Optional metadata JSON output path")
    args = parser.parse_args()

    model_id = _resolve_model_id(args.model)
    output = Path(args.out)
    meta_path = Path(args.meta) if args.meta else None
    export_onnx(
        model_id=model_id,
        output=output,
        input_size=args.input,
        pooling=args.pooling,
        batch_size=args.batch_size,
        opset=args.opset,
        meta_path=meta_path,
    )


if __name__ == "__main__":
    main()
