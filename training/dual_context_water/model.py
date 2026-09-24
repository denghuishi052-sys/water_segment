from __future__ import annotations

from typing import Sequence

import timm
import torch
from torch import nn
from torch.nn import functional as F


class ContextFiLM(nn.Module):
    def __init__(
        self,
        context_channels: int,
        feature_channels: int,
        beta_scale: float | None = None,
        gamma_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.affine = nn.Linear(context_channels, 2 * feature_channels)
        self.beta_scale = beta_scale
        self.gamma_scale = float(gamma_scale)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(context).chunk(2, dim=1)
        gamma = (torch.tanh(gamma) * self.gamma_scale).unsqueeze(-1).unsqueeze(-1)
        if self.beta_scale is not None:
            beta = torch.tanh(beta) * self.beta_scale
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + gamma) + beta


class FpnDecoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        decoder_channels: int = 128,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        def norm_layer(channel_count: int) -> nn.Module:
            if norm == "group":
                groups = min(32, channel_count)
                while channel_count % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, channel_count)
            if norm == "batch":
                return nn.BatchNorm2d(channel_count)
            raise ValueError(f"Unsupported decoder norm: {norm}")

        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1)
            for channel in channels
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                norm_layer(decoder_channels),
                nn.GELU(),
            )
            for _ in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        x = self.lateral[-1](features[-1])
        x = self.smooth[-1](x)
        for index in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.lateral[index](features[index])
            x = self.smooth[index](x)
        return F.interpolate(self.head(x), size=output_size, mode="bilinear", align_corners=False)


class DualContextWaterNet(nn.Module):
    """ONNX-friendly two-input semantic water segmentation network."""

    def __init__(
        self,
        local_backbone: str = "convnext_tiny",
        global_backbone: str = "mobilenetv3_large_100",
        decoder_channels: int = 128,
        pretrained: bool = True,
        context_beta_scale: float | None = None,
        context_gamma_scale: float = 1.0,
        decoder_norm: str = "batch",
    ) -> None:
        super().__init__()
        self.local_encoder = timm.create_model(
            local_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.global_encoder = timm.create_model(
            global_backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=4,
        )
        local_channels = self.local_encoder.feature_info.channels()
        global_channels = self.global_encoder.feature_info.channels()[-1]
        context_channels = 2 * global_channels

        self.context_film = nn.ModuleList(
            ContextFiLM(
                context_channels,
                channel,
                beta_scale=context_beta_scale,
                gamma_scale=context_gamma_scale,
            )
            for channel in local_channels
        )
        self.decoder = FpnDecoder(
            local_channels,
            decoder_channels=decoder_channels,
            norm=decoder_norm,
        )
        self.global_head = nn.Conv2d(global_channels, 1, kernel_size=1)
        self.quality_head = nn.Sequential(
            nn.Linear(context_channels, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _context_vector(global_feature: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        roi = F.interpolate(roi_mask, size=global_feature.shape[-2:], mode="nearest")
        global_average = global_feature.mean(dim=(-2, -1))
        roi_sum = (global_feature * roi).sum(dim=(-2, -1))
        roi_count = roi.sum(dim=(-2, -1)).clamp_min(1.0)
        roi_average = roi_sum / roi_count
        return torch.cat([global_average, roi_average], dim=1)

    def forward(
        self,
        local_image: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features = self.global_encoder(global_context)
        global_feature = global_features[-1]
        roi_mask = global_context[:, 3:4]
        context = self._context_vector(global_feature, roi_mask)

        local_features = self.local_encoder(local_image)
        fused = [
            film(feature, context)
            for film, feature in zip(self.context_film, local_features)
        ]
        local_logits = self.decoder(fused, output_size=local_image.shape[-2:])
        global_logits = F.interpolate(
            self.global_head(global_feature),
            size=global_context.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        quality_logits = self.quality_head(context)
        return local_logits, global_logits, quality_logits
