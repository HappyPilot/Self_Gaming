from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import zmq
from transformers import AutoImageProcessor

# Allow running from either repo layout or docker layout.
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor" / "NitroGen"
if not VENDOR.exists():
    VENDOR = ROOT / "NitroGen"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.inference_session import InferenceSession
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig, NitrogenTokenizer, Tokenizer
from nitrogen.shared import PATH_REPO


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

TRANSFORMERS_PATCHED = False


def _parse_dtype(raw: str) -> torch.dtype:
    key = (raw or "").strip().lower()
    return DTYPE_MAP.get(key, torch.float16)


def _autocast_dtype(dtype: torch.dtype) -> Optional[torch.dtype]:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return None


def _patch_transformers_dtype(dtype: torch.dtype) -> None:
    global TRANSFORMERS_PATCHED
    if TRANSFORMERS_PATCHED:
        return
    if dtype not in (torch.float16, torch.bfloat16):
        return
    from transformers import AutoModel, SiglipVisionModel

    def _wrap(orig):
        def _inner(*args, **kwargs):
            kwargs.setdefault("torch_dtype", dtype)
            kwargs.setdefault("low_cpu_mem_usage", True)
            return orig(*args, **kwargs)
        return _inner

    AutoModel.from_pretrained = _wrap(AutoModel.from_pretrained)
    SiglipVisionModel.from_pretrained = _wrap(SiglipVisionModel.from_pretrained)
    TRANSFORMERS_PATCHED = True


def load_model(checkpoint_path: str, dtype: torch.dtype, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg = ckpt_config.model_cfg
    tokenizer_cfg = ckpt_config.tokenizer_cfg

    print("Checkpoint args:")
    print(json.dumps(ckpt_config.model_dump(), indent=4))

    img_proc = AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)

    if isinstance(model_cfg, NitroGen_Config):
        assert isinstance(
            tokenizer_cfg, NitrogenTokenizerConfig
        ), "NitroGen_Config requires NitrogenTokenizerConfig for tokenization"
        tokenizer_cfg.training = False
        if tokenizer_cfg.game_mapping_cfg is not None:
            tokenizer_cfg.game_mapping_cfg.src_files = [
                x.replace("/mnt/amlfs-02/shared/gaming/gamingvla", str(PATH_REPO))
                for x in tokenizer_cfg.game_mapping_cfg.src_files
            ]
        tokenizer = NitrogenTokenizer(tokenizer_cfg)
        game_mapping = tokenizer.game_mapping
        _patch_transformers_dtype(dtype)
        prev_dtype = torch.get_default_dtype()
        if dtype in (torch.float16, torch.bfloat16):
            torch.set_default_dtype(dtype)
        try:
            model = NitroGen(config=model_cfg, game_mapping=game_mapping)
        finally:
            torch.set_default_dtype(prev_dtype)
        action_downsample_ratio = 1
    else:
        raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

    model = model.to(dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer.eval()
    model.to(device)

    del checkpoint

    return model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio


class InferenceSessionLite(InferenceSession):
    def __init__(self, *args, autocast_dtype: Optional[torch.dtype] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.autocast_dtype = autocast_dtype

    @classmethod
    def from_ckpt(
        cls,
        checkpoint_path: str,
        old_layout=False,
        cfg_scale=1.0,
        context_length=None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio = load_model(
            checkpoint_path, dtype=dtype, device=device
        )

        if game_mapping is not None:
            print("Available games in tokenizer mapping:")
            for game, idx in game_mapping.items():
                print(f"{idx:03d}: {game}")
            selected_game = input("Enter the game ID to use (leave empty for unconditional): ")
            if selected_game == "":
                selected_game = None
            else:
                selected_idx = int(selected_game)
                assert selected_idx in game_mapping.values(), f"Invalid game ID {selected_idx}"
                candidates = [k for k, v in game_mapping.items() if v == selected_idx]
                assert len(candidates) == 1, f"Multiple games found for ID {selected_idx}: {candidates}"
                selected_game = candidates[0]
        else:
            selected_game = None
            print("No game mapping available, proceeding without game conditioning")

        return cls(
            model,
            checkpoint_path,
            tokenizer,
            img_proc,
            ckpt_config,
            game_mapping,
            selected_game,
            old_layout,
            cfg_scale,
            action_downsample_ratio,
            context_length,
            autocast_dtype=_autocast_dtype(dtype),
        )

    def _predict_flowmatching(self, pixel_values, action_tensors):
        available_frames = len(self.obs_buffer)
        frames = torch.zeros(
            (self.max_buffer_size, *pixel_values.shape[1:]),
            dtype=pixel_values.dtype,
            device="cuda",
        )
        frames[-available_frames:] = pixel_values
        dropped_frames = torch.zeros((self.max_buffer_size,), dtype=torch.bool, device="cuda")
        dropped_frames[: self.max_buffer_size - available_frames] = True

        data_with_history = {
            "frames": frames,
            "dropped_frames": dropped_frames,
            "game": self.selected_game,
        }
        tokenized_data_with_history = self.tokenizer.encode(data_with_history)

        frame_mask = torch.ones((self.max_buffer_size,), dtype=torch.bool, device="cuda")
        frame_mask[-1] = False
        data_without_history = {
            "frames": frames,
            "dropped_frames": frame_mask,
            "game": None,
        }
        tokenized_data_without_history = self.tokenizer.encode(data_without_history)

        for tokenized_data in [tokenized_data_with_history, tokenized_data_without_history]:
            for k, v in tokenized_data.items():
                if isinstance(v, torch.Tensor):
                    tokenized_data[k] = v.unsqueeze(0).to("cuda")
                elif isinstance(v, np.ndarray):
                    tokenized_data[k] = torch.tensor(v, device="cuda").unsqueeze(0)
                else:
                    tokenized_data[k] = [v]

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                if self.cfg_scale == 1.0:
                    model_output = self.model.get_action(
                        tokenized_data_with_history, old_layout=self.old_layout
                    )
                else:
                    model_output = self.model.get_action_with_cfg(
                        tokenized_data_with_history,
                        tokenized_data_without_history,
                        cfg_scale=self.cfg_scale,
                    )
                predicted_actions = self.tokenizer.decode(model_output)

        return predicted_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="NitroGen inference server (fp16/bf16 aware)")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length")
    parser.add_argument("--dtype", type=str, default=os.getenv("NITROGEN_DTYPE", "fp16"))
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NitroGen inference on this server.")
    device = "cuda"

    session = InferenceSessionLite.from_ckpt(
        args.ckpt,
        old_layout=args.old_layout,
        cfg_scale=args.cfg,
        context_length=args.ctx,
        dtype=dtype,
        device=device,
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Server running on port {args.port}")
    print(f"Device: {device}, dtype: {args.dtype}")
    print(f"Waiting for requests...")
    print(f"{'='*60}\n")

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                request = socket.recv()
                request = pickle.loads(request)
                if request["type"] == "reset":
                    session.reset()
                    response = {"status": "ok"}
                    print("Session reset")
                elif request["type"] == "info":
                    info = session.info()
                    response = {"status": "ok", "info": info}
                    print("Sent session info")
                elif request["type"] == "predict":
                    raw_image = request["image"]
                    result = session.predict(raw_image)
                    response = {"status": "ok", "pred": result}
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}
                socket.send(pickle.dumps(response))
    except KeyboardInterrupt:
        print("\nShutting down server...")
        raise SystemExit(0)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
