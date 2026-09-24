from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_context_water import DualContextDataset, DualContextWaterNet


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    negative_weight: float = 1.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.45,
    tversky_beta: float = 0.55,
) -> torch.Tensor:
    pixel_weight = target + float(negative_weight) * (1.0 - target)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * pixel_weight
    ).mean(dim=(-3, -2, -1))
    dice = dice_loss(logits, target).mean(dim=1)
    weight = sample_weight.flatten()
    probability = torch.sigmoid(logits)
    true_positive = (probability * target).sum(dim=(-3, -2, -1))
    false_positive = (probability * (1.0 - target)).sum(dim=(-3, -2, -1))
    false_negative = ((1.0 - probability) * target).sum(dim=(-3, -2, -1))
    tversky = 1.0 - (true_positive + 1.0) / (
        true_positive
        + float(tversky_alpha) * false_positive
        + float(tversky_beta) * false_negative
        + 1.0
    )
    per_sample = bce + dice + float(tversky_weight) * tversky
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target_edge = F.max_pool2d(target, 3, stride=1, padding=1) - (
        -F.max_pool2d(-target, 3, stride=1, padding=1)
    )
    probability_edge = F.max_pool2d(probability, 3, stride=1, padding=1) - (
        -F.max_pool2d(-probability, 3, stride=1, padding=1)
    )
    per_sample = (probability_edge - target_edge).abs().mean(dim=(-3, -2, -1))
    weight = sample_weight.flatten()
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    intersection = union = predicted = target_count = 0.0
    for batch in loader:
        local = batch["local_image"].to(device, non_blocking=True)
        context = batch["global_context"].to(device, non_blocking=True)
        target = batch["local_mask"].to(device, non_blocking=True)
        logits, _, _ = model(local, context)
        prediction = torch.sigmoid(logits) >= 0.5
        truth = target >= 0.5
        intersection += float((prediction & truth).sum())
        union += float((prediction | truth).sum())
        predicted += float(prediction.sum())
        target_count += float(truth.sum())
    iou = intersection / max(union, 1.0)
    dice = 2.0 * intersection / max(predicted + target_count, 1.0)
    precision = intersection / max(predicted, 1.0)
    recall = intersection / max(target_count, 1.0)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dual_context_water.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DualContextDataset(ROOT / config["train_manifest"], augment=True)
    val_dataset = DualContextDataset(ROOT / config["val_manifest"], augment=False)
    if args.smoke:
        base_rows = [
            row for row in train_dataset.rows if row.get("domain", "base") != "satellite"
        ][:4]
        satellite_rows = [
            row for row in train_dataset.rows if row.get("domain", "base") == "satellite"
        ][:4]
        train_dataset.rows = base_rows + satellite_rows
        val_dataset.rows = val_dataset.rows[:4]
    sampler = None
    satellite_fraction = config.get("satellite_batch_fraction")
    if satellite_fraction is not None:
        target_fraction = float(satellite_fraction)
        domains = [row.get("domain", "base") for row in train_dataset.rows]
        satellite_count = sum(domain == "satellite" for domain in domains)
        base_count = len(domains) - satellite_count
        if satellite_count <= 0 or base_count <= 0:
            raise ValueError("Domain balancing requires both satellite and base samples")
        weights = [
            target_fraction / satellite_count
            if domain == "satellite"
            else (1.0 - target_fraction) / base_count
            for domain in domains
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["workers"]),
        pin_memory=True,
        persistent_workers=int(config["workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["workers"]),
        pin_memory=True,
        persistent_workers=int(config["workers"]) > 0,
    )

    model = DualContextWaterNet(
        local_backbone=config["local_backbone"],
        global_backbone=config["global_backbone"],
        decoder_channels=int(config["decoder_channels"]),
        pretrained=bool(config["pretrained"]),
        context_beta_scale=config.get("context_beta_scale"),
        context_gamma_scale=float(config.get("context_gamma_scale", 1.0)),
        decoder_norm=config.get("decoder_norm", "batch"),
    ).to(device)
    if bool(config.get("gradient_checkpointing", False)):
        for encoder in (model.local_encoder, model.global_encoder):
            if hasattr(encoder, "set_grad_checkpointing"):
                encoder.set_grad_checkpointing(True)
    initial_metrics: dict[str, float] = {}
    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(ROOT / init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        initial_metrics = {
            key: float(value)
            for key, value in checkpoint.get("metrics", {}).items()
        }
        print(
            f"Loaded initial checkpoint: {ROOT / init_checkpoint} "
            f"(metrics={initial_metrics})",
            flush=True,
        )
        if bool(config.get("reset_decoder_batchnorm", False)):
            for module in model.decoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.reset_running_stats()
            print("Reset decoder BatchNorm running statistics.", flush=True)
    else:
        partial_checkpoint = config.get("partial_init_checkpoint")
        if partial_checkpoint:
            checkpoint = torch.load(
                ROOT / partial_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            source_state = checkpoint["model"]
            target_state = model.state_dict()
            loaded = 0
            adapted = 0
            for name, target in list(target_state.items()):
                candidates = [name]
                if name.startswith("global_encoder."):
                    candidates.append("local_encoder." + name.removeprefix("global_encoder."))
                for candidate in candidates:
                    source = source_state.get(candidate)
                    if source is None:
                        continue
                    if source.shape == target.shape:
                        target_state[name] = source
                        loaded += 1
                        break
                    if (
                        source.ndim == 4
                        and target.ndim == 4
                        and source.shape[0] == target.shape[0]
                        and source.shape[1] == 3
                        and target.shape[1] == 4
                        and source.shape[2:] == target.shape[2:]
                    ):
                        expanded = target.clone()
                        expanded[:, :3] = source
                        expanded[:, 3:4] = source.mean(dim=1, keepdim=True)
                        target_state[name] = expanded
                        loaded += 1
                        adapted += 1
                        break
            model.load_state_dict(target_state)
            print(
                f"Partially loaded {loaded} tensors from {ROOT / partial_checkpoint} "
                f"({adapted} adapted to four channels).",
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if str(config.get("amp_dtype", "float16")).lower() == "bfloat16"
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    accumulation = max(int(config["gradient_accumulation"]), 1)
    freeze_encoder_epochs = int(config.get("freeze_encoder_epochs", 0))
    if freeze_encoder_epochs > 0:
        for encoder in (model.local_encoder, model.global_encoder):
            encoder.requires_grad_(False)
        print(f"Encoders frozen for {freeze_encoder_epochs} epoch(s).", flush=True)

    output_dir = ROOT / config["output_dir"]
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "results.csv"
    best_dice = (
        -1.0
        if bool(config.get("reset_best_metric", False))
        else float(initial_metrics.get("dice", -1.0))
    )
    stale_epochs = 0
    fields = ["epoch", "seconds", "train_loss", "iou", "dice", "precision", "recall", "learning_rate"]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    if init_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "epoch": 0,
                "metrics": initial_metrics,
            },
            output_dir / "best.pt",
        )

    training_started = time.perf_counter()
    max_hours = float(config.get("max_hours", 0.0))
    for epoch in range(1, epochs + 1):
        if max_hours > 0 and time.perf_counter() - training_started >= max_hours * 3600:
            print(f"Time limit reached before epoch {epoch}; stopping.", flush=True)
            break
        if epoch == freeze_encoder_epochs + 1 and freeze_encoder_epochs > 0:
            for encoder in (model.local_encoder, model.global_encoder):
                encoder.requires_grad_(True)
            print("Encoders unfrozen.", flush=True)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            local = batch["local_image"].to(device, non_blocking=True)
            context = batch["global_context"].to(device, non_blocking=True)
            local_target = batch["local_mask"].to(device, non_blocking=True)
            global_target = batch["global_mask"].to(device, non_blocking=True)
            quality_target = batch["quality_target"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                local_logits, global_logits, quality_logits = model(local, context)
                negative_weight = float(config.get("false_positive_weight", 1.0))
                loss = segmentation_loss(
                    local_logits,
                    local_target,
                    sample_weight,
                    negative_weight,
                    float(config.get("tversky_weight", 0.0)),
                    float(config.get("tversky_alpha", 0.45)),
                    float(config.get("tversky_beta", 0.55)),
                )
                loss = loss + float(config["global_loss_weight"]) * segmentation_loss(
                    global_logits,
                    global_target,
                    sample_weight,
                    negative_weight,
                    float(config.get("tversky_weight", 0.0)),
                    float(config.get("tversky_alpha", 0.45)),
                    float(config.get("tversky_beta", 0.55)),
                )
                loss = loss + float(config.get("boundary_loss_weight", 0.0)) * boundary_loss(
                    local_logits, local_target, sample_weight
                )
                quality_loss = F.binary_cross_entropy_with_logits(
                    quality_logits, quality_target, reduction="none"
                ).mean(dim=1)
                weight = sample_weight.flatten()
                loss = loss + float(config["quality_loss_weight"]) * (
                    (quality_loss * weight).sum() / weight.sum().clamp_min(1e-6)
                )
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, step={step}: {float(loss.detach())}"
                )
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.get("gradient_clip_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())

        metrics = validate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 2),
            "train_loss": running_loss / max(len(train_loader), 1),
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(row, flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
            minimum_epochs = int(config.get("min_epochs", 0))
            if epoch >= minimum_epochs and stale_epochs >= int(config["patience"]):
                print(f"Early stopping at epoch {epoch}; best_dice={best_dice:.6f}", flush=True)
                break


if __name__ == "__main__":
    main()
