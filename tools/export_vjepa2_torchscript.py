#!/usr/bin/env python3
"""Export V-JEPA2 ViT-L encoder to TorchScript for vl_jepa_agent."""
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
    from transformers import AutoImageProcessor, AutoModel
except Exception as exc:  # pragma: no cover - manual script
    raise SystemExit(
        "Failed to import transformers. Install first: pip install transformers -- reason: %s" % exc
    )

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("export_vjepa2_torchscript")

DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_INPUT_SIZE = 256
DEFAULT_OUTPUT = "vjepa2_vitl_256.ts"
DEFAULT_POOLING = "mean"


def _resolve_device(device: str) -> str:
    if device and device.lower() not in {"auto", "default"}:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(
        self,
        model: nn.Module,
        input_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        pooling: str,
        frames: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_size = int(input_size)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.pooling = pooling
        self.frames = max(1, int(frames))
        signature = inspect.signature(self.model.forward)
        self._use_pixel_values = "pixel_values" in signature.parameters
        self._use_pixel_values_videos = "pixel_values_videos" in signature.parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.mean.dtype)
        if self._use_pixel_values_videos:
            if x.ndim == 4:
                x = x.unsqueeze(1).repeat(1, self.frames, 1, 1, 1)
            if x.ndim != 5:
                raise RuntimeError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")
            bsz, frames, channels, height, width = x.shape
            x = x.view(bsz * frames, channels, height, width)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            x = x.view(bsz, frames, channels, self.input_size, self.input_size)
            mean = self.mean.unsqueeze(1)
            std = self.std.unsqueeze(1)
            x = (x - mean) / std
            outputs = self.model(pixel_values_videos=x)
        else:
            if x.ndim != 4:
                raise RuntimeError(f"Expected 4D input, got shape {tuple(x.shape)}")
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            if self._use_pixel_values:
                outputs = self.model(pixel_values=x)
            else:
                outputs = self.model(x)
        return _extract_embedding(outputs, self.pooling)


def export_torchscript(
    model_id: str,
    output: Path,
    input_size: int,
    device: str,
    fp16: bool,
    pooling: str,
    batch_size: int,
    frames: int,
    meta_path: Optional[Path],
) -> Path:
    output = output.expanduser()
    if output.is_dir():
        output = output / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = meta_path or output.with_suffix(".meta.json")

    device = _resolve_device(device)
    logger.info("Loading model %s (device=%s)", model_id, device)
    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Image processor not found for %s (%s); using default mean/std.", model_id, exc)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model.to(device)

    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406]) if processor else [0.485, 0.456, 0.406]
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225]) if processor else [0.229, 0.224, 0.225]

    wrapper = EncoderWrapper(model, input_size, mean, std, pooling, frames)
    wrapper.eval().to(device)
    if fp16 and device.startswith("cuda"):
        wrapper.half()

    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    example = torch.zeros((batch_size, 3, input_size, input_size), device=device, dtype=dtype)
    with torch.inference_mode():
        output_tensor = wrapper(example)
    embed_dim = int(output_tensor.shape[-1])
    logger.info("Embedding shape: %s (dim=%s)", tuple(output_tensor.shape), embed_dim)

    traced = torch.jit.trace(wrapper, example, strict=False)
    traced.save(str(output))

    meta = {
        "model_id": model_id,
        "input_size": input_size,
        "pooling": pooling,
        "frames": frames,
        "embed_dim": embed_dim,
        "dtype": str(dtype).replace("torch.", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved TorchScript: %s", output)
    logger.info("Saved metadata: %s", meta_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V-JEPA2 encoder to TorchScript")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .ts path or directory")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Input size (default: 256)")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, cuda:0)")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 (CUDA only)")
    parser.add_argument("--pooling", choices=["mean", "cls"], default=DEFAULT_POOLING, help="Token pooling")
    parser.add_argument("--batch-size", type=int, default=1, help="Trace batch size")
    parser.add_argument("--frames", type=int, default=2, help="Frames per clip for export (default: 2)")
    parser.add_argument("--meta", help="Optional metadata JSON output path")
    args = parser.parse_args()

    output = Path(args.output)
    meta_path = Path(args.meta) if args.meta else None
    export_torchscript(
        model_id=args.model_id,
        output=output,
        input_size=args.input_size,
        device=args.device,
        fp16=args.fp16,
        pooling=args.pooling,
        batch_size=args.batch_size,
        frames=args.frames,
        meta_path=meta_path,
    )


if __name__ == "__main__":
    main()
