$ErrorActionPreference = "Stop"

$Python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe"
$MetricsCsv = ".\runs\eval_yolov8m_b8_3_test\metrics_test.csv"
$Data = ".\data\processed\waterlogging.yaml"
$HardDir = ".\data\hard_balanced"
$Config = ".\configs\train_hard_balanced_freeze.yaml"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Python .\scripts\09_create_balanced_finetune_dataset.py `
  --metrics_csv $MetricsCsv `
  --data $Data `
  --output_dir $HardDir `
  --total_samples 100 `
  --iou_threshold 0.4 `
  --val_ratio 0.15 `
  --seed 42 `
  --source_split test

& $Python .\scripts\08_finetune_hard_samples.py --config $Config
