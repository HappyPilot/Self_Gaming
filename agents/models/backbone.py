"""Shared dual-branch backbone for Orin-Gamer Swarm.

This module exposes the Backbone class which consumes a downscaled RGB frame
and a 128-D non-visual feature vector (numeric stats + OCR/object summaries)
then produces a fused embedding that downstream PPO heads can use.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Tiny CNN + MLP backbone that outputs a shared latent state."""

    def __init__(
        self,
        frame_shape: Sequence[int] = (3, 96, 54),
        non_visual_dim: int = 128,
        vision_embed_dim: int = 384,
        non_visual_embed_dim: int = 128,
        final_dim: int = 512,
    ) -> None:
        super().__init__()
        if len(frame_shape) != 3:
            raise ValueError("frame_shape must be (C, H, W)")
        self.frame_shape: Tuple[int, int, int] = tuple(frame_shape)  # type: ignore[assignment]
        self.non_visual_dim = non_visual_dim

        in_channels = self.frame_shape[0]
        self.vision_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, vision_embed_dim),
            nn.ReLU(inplace=True),
        )

        self.non_visual_branch = nn.Sequential(
            nn.Linear(non_visual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, non_visual_embed_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = vision_embed_dim + non_visual_embed_dim
        if final_dim and final_dim != fused_dim:
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, final_dim),
                nn.ReLU(inplace=True),
            )
            self.output_dim = final_dim
        else:
            self.fusion = nn.Identity()
            self.output_dim = fused_dim

    def forward(
        self,
        frame_tensor: Optional[torch.Tensor],
        non_visual_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused latent state for PPO heads."""

        if frame_tensor is None:
            batch = non_visual_tensor.size(0)
            frame_tensor = torch.zeros(
                batch,
                *self.frame_shape,
                device=non_visual_tensor.device,
                dtype=non_visual_tensor.dtype,
            )
        if frame_tensor.ndim != 4:
            raise ValueError("frame_tensor must be [B, C, H, W]")
        if non_visual_tensor.ndim != 2 or non_visual_tensor.size(1) != self.non_visual_dim:
            raise ValueError(
                f"non_visual_tensor must be [B, {self.non_visual_dim}] but got {tuple(non_visual_tensor.shape)}"
            )

        vision_feat = self.vision_branch(frame_tensor)
        vision_emb = self.vision_proj(self.global_pool(vision_feat))

        non_visual_emb = self.non_visual_branch(non_visual_tensor)
        fused = torch.cat([vision_emb, non_visual_emb], dim=1)
        return self.fusion(fused)
