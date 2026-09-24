# SAM3-LoRA + Proposal Selector Runbook

The production default remains YOLO-only. The test split must stay sealed
until the adapter, selector, and thresholds are frozen.

## 1. Build the train/val manifest

```powershell
python scripts/15_build_sam3_finetune_manifest.py
```

This emits `data/sam3_finetune/manifest_train_val.jsonl`, validates
`source_stem` isolation across train/val/test, and never reads test images.
The weighted sampler targets 40% positive/context, 20% boundary, 25%
hard-negative, and 15% small-object recovery examples.

## 2. Train the Base SAM3 selector

```powershell
python scripts/16_cache_sam3_proposals.py
python scripts/17_train_sam3_selector.py
```

The cache contains YOLO, text+box, interactive-points, and global masks plus
their fusion features and true `delta_iou`. The selector is exported as plain
JSON. Its validation threshold maximizes IoU while keeping precision within
one percentage point of YOLO.

## 3. Train Q/V LoRA

```powershell
python scripts/18_train_sam3_lora.py
```

The trainer freezes every base parameter, wraps the 32 fused visual `qkv`
layers, and adds rank-4 updates to Q and V only. Defaults are alpha 16,
dropout 0.05, crop size 512, batch 1, accumulation 8, FP16, AdamW at `1e-4`,
20 epochs, and patience 5. The official matcher and Dice implementation are
used with BCE and presence losses.

## 4. Regenerate proposals and retrain the selector

Set these fields in a copied trial config:

```yaml
sam3_adapter_enabled: true
sam3_lora_path: runs/sam3_finetune/lora/best
```

Then rerun scripts 16 and 17 with separate LoRA cache/output paths. Do not
reuse the Base SAM3 selector.

## 5. Freeze and evaluate

Enable the chosen adapter and selector only in the frozen evaluation config:

```yaml
sam3_adapter_enabled: true
sam3_selector_enabled: true
sam3_selector_path: runs/sam3_finetune/lora_selector.json
```

Run the sealed test exactly once. If any acceptance criterion fails, leave
both flags disabled and keep YOLO-only in production.
