$ErrorActionPreference = "Stop"

$Model = ".\runs\segment\runs\segment\waterlogging_yolov8m_640_b8-3\weights\best.pt"
$Data = ".\data\processed\waterlogging.yaml"
$OutputDir = ".\runs\eval_yolov8m_b8_3_test"
$Python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe"

& $Python .\scripts\05_eval_test.py `
  --model $Model `
  --data $Data `
  --split test `
  --imgsz 704 `
  --device 0 `
  --conf 0.25 `
  --iou 0.5 `
  --mask_mode auto `
  --foreground_rgb "128,0,0" `
  --tolerance 10 `
  --output_dir $OutputDir `
  --visual_num 100
