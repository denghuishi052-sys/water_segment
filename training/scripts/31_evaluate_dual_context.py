from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_context_water import DualContextDataset, DualContextWaterNet


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = DualContextWaterNet(
        local_backbone=config["local_backbone"],
        global_backbone=config["global_backbone"],
        decoder_channels=int(config["decoder_channels"]),
        pretrained=False,
        context_beta_scale=config.get("context_beta_scale"),
        context_gamma_scale=float(config.get("context_gamma_scale", 1.0)),
        decoder_norm=config.get("decoder_norm", "batch"),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = DualContextDataset(ROOT / args.manifest, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    intersection = union = predicted = target_count = 0.0
    for batch in loader:
        local = batch["local_image"].to(device, non_blocking=True)
        context = batch["global_context"].to(device, non_blocking=True)
        target = batch["local_mask"].to(device, non_blocking=True) >= 0.5
        logits, _, _ = model(local, context)
        prediction = torch.sigmoid(logits) >= 0.5
        intersection += float((prediction & target).sum())
        union += float((prediction | target).sum())
        predicted += float(prediction.sum())
        target_count += float(target.sum())
    metrics = {
        "images": len(dataset),
        "iou": intersection / max(union, 1.0),
        "dice": 2.0 * intersection / max(predicted + target_count, 1.0),
        "precision": intersection / max(predicted, 1.0),
        "recall": intersection / max(target_count, 1.0),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
