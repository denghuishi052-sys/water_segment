"""Differentiable SAM 3 Q/V-LoRA training loop."""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.sam3_lora.data import read_jsonl
from training.sam3_lora.lora import (
    inject_vision_qv_lora,
    save_lora_adapter,
    trainable_parameters,
)


class ManifestDataset(Dataset):
    def __init__(self, records: list[dict], crop_size: int = 512) -> None:
        self.records = records
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(record["image"])
        original_height, original_width = mask.shape
        box = record["gt_box_xyxy"]
        if box is not None:
            scale_x = self.crop_size / float(original_width)
            scale_y = self.crop_size / float(original_height)
            box = [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1),
            "mask": torch.from_numpy((mask > 0).astype(np.float32)),
            "prompt": record["prompt"],
            "gt_box_xyxy": box,
            "record": record,
        }


def collate_single(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("SAM 3 LoRA v1 requires batch_size=1")
    return batch[0]


def _normalized_cxcywh(box, width: int, height: int) -> torch.Tensor:
    x1, y1, x2, y2 = (float(value) for value in box)
    return torch.tensor(
        [
            ((x1 + x2) * 0.5) / width,
            ((y1 + y2) * 0.5) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ],
        dtype=torch.float32,
    )


class DifferentiableSam3:
    def __init__(self, model, device: str, resolution: int = 1008) -> None:
        from sam3.model.data_misc import FindStage

        self.model = model
        self.device = torch.device(device)
        self.resolution = int(resolution)
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device, non_blocking=True).float().unsqueeze(0)
        image = F.interpolate(
            image,
            (self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )
        return image.div(255.0).sub(0.5).div(0.5)

    def forward(self, image, prompt: str, box_xyxy=None):
        tensor = self.prepare_image(image)
        backbone_out = self.model.backbone.forward_image(tensor)
        backbone_out.update(
            self.model.backbone.forward_text([prompt], device=self.device)
        )
        geometric = self.model._get_dummy_prompt()
        if box_xyxy is not None:
            box = _normalized_cxcywh(
                box_xyxy, int(image.shape[-1]), int(image.shape[-2])
            ).to(self.device)
            geometric.append_boxes(
                box.view(1, 1, 4),
                torch.ones((1, 1), dtype=torch.bool, device=self.device),
            )
        return self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            find_target=None,
            geometric_prompt=geometric,
        )


def matched_segmentation_loss(model, outputs: dict, target: torch.Tensor) -> dict:
    from sam3.train.loss.loss_fns import dice_loss

    target = target.float().to(outputs["pred_masks"].device)
    positive = bool(target.any())
    pred_masks = outputs["pred_masks"]
    pred_logits = outputs["pred_logits"]
    presence = outputs["presence_logit_dec"].reshape(-1)
    presence_target = torch.ones_like(presence) if positive else torch.zeros_like(
        presence
    )
    presence_loss = F.binary_cross_entropy_with_logits(
        presence, presence_target
    )
    target_resized = F.interpolate(
        target[None, None],
        pred_masks.shape[-2:],
        mode="nearest",
    )[0, 0]
    if positive:
        coordinates = torch.nonzero(target > 0, as_tuple=False)
        y1, x1 = coordinates.min(dim=0).values
        y2, x2 = coordinates.max(dim=0).values + 1
        box = torch.tensor(
            [
                ((x1 + x2).float() * 0.5) / target.shape[1],
                ((y1 + y2).float() * 0.5) / target.shape[0],
                (x2 - x1).float() / target.shape[1],
                (y2 - y1).float() / target.shape[0],
            ],
            device=target.device,
        ).view(1, 4)
        _, source_indices, _ = model.matcher(
            {"pred_logits": pred_logits, "pred_boxes": outputs["pred_boxes"]},
            {
                "boxes": box,
                "boxes_padded": box.view(1, 1, 4),
                "num_boxes": torch.ones(
                    1, dtype=torch.long, device=target.device
                ),
            },
        )
        query = int(source_indices[0])
    else:
        query = int(pred_logits.sigmoid().reshape(-1).argmax())
    selected = pred_masks[0, query]
    bce = F.binary_cross_entropy_with_logits(selected, target_resized)
    dice = dice_loss(
        selected[None], target_resized.reshape(1, -1), num_boxes=1.0
    )
    total = bce + dice + presence_loss
    return {
        "loss": total,
        "bce": bce.detach(),
        "dice": dice.detach(),
        "presence": presence_loss.detach(),
    }


@torch.no_grad()
def validate(forwarder: DifferentiableSam3, loader: DataLoader) -> dict:
    tp = fp = fn = 0
    for batch in loader:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=forwarder.device.type == "cuda",
        ):
            outputs = forwarder.forward(
                batch["image"], batch["prompt"], batch["gt_box_xyxy"]
            )
        scores = outputs["pred_logits"].sigmoid().reshape(-1)
        scores = scores * outputs["presence_logit_dec"].sigmoid().reshape(-1)[0]
        query = int(scores.argmax())
        pred = outputs["pred_masks"][0, query]
        pred = F.interpolate(
            pred[None, None],
            batch["mask"].shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].sigmoid() > 0.5
        gt = batch["mask"].to(pred.device) > 0
        tp += int(torch.logical_and(pred, gt).sum())
        fp += int(torch.logical_and(pred, ~gt).sum())
        fn += int(torch.logical_and(~pred, gt).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def train_lora(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.05,
    crop_size: int = 512,
    accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    epochs: int = 20,
    patience: int = 5,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 0,
) -> dict:
    from sam3 import build_sam3_image_model
    from sam3.model.vitdet import Mlp

    if str(device).startswith("cuda"):
        # cuDNN SDPA backward can fail on some Windows + RTX 3060 shapes with
        # "No execution plans support the graph". Keep Flash/mem-efficient
        # paths available to stay within 12 GB, but exclude cuDNN's planner.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def differentiable_mlp_forward(module, value):
        value = module.fc1(value)
        value = module.act(value)
        value = module.drop1(value)
        value = module.norm(value)
        value = module.fc2(value)
        return module.drop2(value)

    # The distributed SAM 3 inference package swaps this path for a fused
    # no-grad kernel. Restore the mathematically equivalent autograd path for
    # LoRA training without modifying the installed package.
    Mlp.forward = differentiable_mlp_forward

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = read_jsonl(manifest)
    if any(record["split"] == "test" for record in records):
        raise ValueError("Test records are forbidden during LoRA training")
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    if max_train_samples:
        train_records = train_records[:max_train_samples]
    if max_val_samples:
        val_records = val_records[:max_val_samples]
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
        eval_mode=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    # SAM 3's inference-mode fused MLP explicitly disables autograd. Keep
    # submodules in their differentiable training paths, while suppressing the
    # top-level automatic matching because this loop invokes the official
    # matcher explicitly with our compact targets.
    model.train()
    model.training = False
    target_modules = inject_vision_qv_lora(
        model, rank=rank, alpha=alpha, dropout=dropout
    )
    from training.sam3_lora.lora import LoRAQKVLinear

    for module in model.modules():
        if isinstance(module, LoRAQKVLinear):
            module.train()
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not any(token in name for token in (".q_a.", ".q_b.", ".v_a.", ".v_b."))
        for name in trainable_names
    ):
        raise AssertionError("Only Q/V LoRA parameters may be trainable")
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    train_dataset = ManifestDataset(train_records, crop_size)
    sampler = WeightedRandomSampler(
        [float(record["sample_weight"]) for record in train_records],
        num_samples=len(train_records),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_single,
        num_workers=0,
    )
    val_loader = DataLoader(
        ManifestDataset(val_records, crop_size),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
        num_workers=0,
    )
    forwarder = DifferentiableSam3(model, device)
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    baseline = validate(forwarder, val_loader)
    best_iou = -1.0
    stale = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = forwarder.forward(
                    batch["image"], batch["prompt"], batch["gt_box_xyxy"]
                )
                losses = matched_segmentation_loss(
                    model, outputs, batch["mask"]
                )
                loss = losses["loss"] / accumulation_steps
            scaler.scale(loss).backward()
            running += float(losses["loss"].detach())
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = validate(forwarder, val_loader)
        metrics.update(
            {"epoch": epoch, "train_loss": running / max(len(train_loader), 1)}
        )
        history.append(metrics)
        precision_ok = metrics["precision"] >= baseline["precision"] - 0.01
        if precision_ok and metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            stale = 0
            save_lora_adapter(
                model,
                output / "best",
                base_checkpoint=checkpoint,
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                extra_metadata={
                    "crop_size": crop_size,
                    "model_input_resolution": 1008,
                    "validation_metrics": metrics,
                    "baseline_validation_metrics": baseline,
                },
            )
        else:
            stale += 1
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if stale >= patience:
            break
    return {
        "baseline": baseline,
        "best_iou": best_iou,
        "epochs_completed": len(history),
        "target_modules": target_modules,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated()) / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    }
