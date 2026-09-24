"""SAM 3 LoRA and proposal-selector training support.

Submodules are intentionally not imported here. This keeps data-manifest and
selector utilities lightweight and avoids loading PyTorch into YOLO/OpenCV
processes that do not need LoRA.
"""
