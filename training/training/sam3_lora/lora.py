"""Minimal Q/V-sliced LoRA for SAM 3 fused vision-attention QKV layers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class LoRAQKVLinear(nn.Module):
    """Wrap ``Linear(C, 3C)`` and add low-rank updates to Q and V only."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if base.out_features != base.in_features * 3:
            raise ValueError("LoRAQKVLinear requires a fused C -> 3C projection")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        width = base.in_features
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.q_a = nn.Linear(
            width, self.rank, bias=False, **factory_kwargs
        )
        self.q_b = nn.Linear(
            self.rank, width, bias=False, **factory_kwargs
        )
        self.v_a = nn.Linear(
            width, self.rank, bias=False, **factory_kwargs
        )
        self.v_b = nn.Linear(
            self.rank, width, bias=False, **factory_kwargs
        )
        nn.init.kaiming_uniform_(self.q_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_b.weight)
        nn.init.zeros_(self.v_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        dropped = self.dropout(inputs)
        q_delta = self.q_b(self.q_a(dropped)) * self.scaling
        v_delta = self.v_b(self.v_a(dropped)) * self.scaling
        zeros = torch.zeros_like(q_delta)
        return base_output + torch.cat((q_delta, zeros, v_delta), dim=-1)


def _parent_and_name(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_vision_qv_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_names: Iterable[str] | None = None,
) -> list[str]:
    """Freeze the model and wrap fused vision ``qkv`` projections."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    requested = set(target_names or ())
    candidates = []
    for name, module in model.named_modules():
        if not name.endswith(".qkv") or not isinstance(module, nn.Linear):
            continue
        if module.out_features != module.in_features * 3:
            continue
        # SAM 3 ViT modules are under the visual backbone; decoder projections
        # use separate q/k/v linears and are intentionally excluded in v1.
        if "backbone" not in name and "visual" not in name:
            continue
        if requested and name not in requested:
            continue
        candidates.append(name)
    if requested - set(candidates):
        raise ValueError(
            f"LoRA target modules not found: {sorted(requested - set(candidates))}"
        )
    if not candidates:
        raise RuntimeError("No fused SAM 3 vision qkv modules matched LoRA")
    for name in candidates:
        parent, attribute = _parent_and_name(model, name)
        setattr(
            parent,
            attribute,
            LoRAQKVLinear(
                getattr(parent, attribute),
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
    return candidates


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def checkpoint_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
    extra_metadata: dict | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(
            token in name
            for token in (".q_a.weight", ".q_b.weight", ".v_a.weight", ".v_b.weight")
        )
    }
    torch.save(state, output / "adapter.pt")
    metadata = {
        "format": "sam3_qv_lora_v1",
        "base_checkpoint": str(Path(base_checkpoint)),
        "base_checkpoint_sha256": checkpoint_fingerprint(base_checkpoint),
        "target_modules": target_modules,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
    }
    metadata.update(extra_metadata or {})
    (output / "adapter_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def load_lora_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    strict_fingerprint: bool = True,
) -> dict:
    adapter = Path(adapter_dir)
    metadata = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != "sam3_qv_lora_v1":
        raise ValueError("Unsupported SAM 3 adapter format")
    if strict_fingerprint:
        actual = checkpoint_fingerprint(base_checkpoint)
        if actual != metadata["base_checkpoint_sha256"]:
            raise ValueError("LoRA base checkpoint fingerprint mismatch")
    injected = inject_vision_qv_lora(
        model,
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        dropout=float(metadata.get("dropout", 0.0)),
        target_names=metadata["target_modules"],
    )
    state = torch.load(
        adapter / "adapter.pt", map_location="cpu", weights_only=True
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if name in state]
    missing_lora = [
        name
        for name in state
        if name not in model.state_dict()
    ]
    if unexpected or missing_lora:
        raise ValueError(
            f"Invalid LoRA state: unexpected={unexpected}, missing={missing_lora}"
        )
    metadata["loaded_target_modules"] = injected
    return metadata
