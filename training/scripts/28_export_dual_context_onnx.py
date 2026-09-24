from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_context_water import DualContextWaterNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/dual_context_water/convnext_tiny_mobilenetv3/best.pt",
    )
    parser.add_argument(
        "--output",
        default="onnx/dual_context_water_convnext_tiny_1024.onnx",
    )
    parser.add_argument("--local_size", type=int, default=1024)
    parser.add_argument("--global_size", type=int, default=512)
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
    model.eval()
    local = torch.zeros(1, 3, args.local_size, args.local_size)
    context = torch.zeros(1, 4, args.global_size, args.global_size)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (local, context),
        output,
        input_names=["local_image", "global_context"],
        output_names=["water_logits", "global_logits", "quality_logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "local_image": np.zeros((1, 3, args.local_size, args.local_size), dtype=np.float32),
            "global_context": np.zeros((1, 4, args.global_size, args.global_size), dtype=np.float32),
        },
    )
    print(f"ONNX validated: {output}")
    print(f"output shapes: {[item.shape for item in outputs]}")


if __name__ == "__main__":
    main()
