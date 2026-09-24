$ErrorActionPreference = "Stop"

$MetricsCsv = ".\runs\eval_yolov8m_b8_3_test\metrics_test.csv"
$Data = ".\data\processed\waterlogging.yaml"
$HardDir = ".\data\hard"
$Config = ".\configs\train_hard_finetune.yaml"
$Python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Python .\scripts\07_hard_sample_mining.py `
  --metrics_csv $MetricsCsv `
  --data $Data `
  --output_dir $HardDir `
  --iou_threshold 0.4 `
  --normal_ratio 0.3 `
  --source_split test

& $Python .\scripts\08_finetune_hard_samples.py --config $Config
