$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

Write-Output '== Repository =='
Write-Output ("Root:   {0}" -f $repoRoot)
Write-Output ("Branch: {0}" -f (git branch --show-current))
git remote -v

Write-Output ''
Write-Output '== Working tree summary =='
$status = @(git status --porcelain=v1 -uall)
[PSCustomObject]@{
    Total     = $status.Count
    Modified  = @($status | Where-Object { $_ -match '^ M|^M |^MM' }).Count
    Added     = @($status | Where-Object { $_ -match '^A |^ A' }).Count
    Deleted   = @($status | Where-Object { $_ -match '^ D|^D ' }).Count
    Untracked = @($status | Where-Object { $_ -match '^\?\?' }).Count
} | Format-List

Write-Output '== Tracked generated content =='
$tracked = @(git ls-files)
foreach ($prefix in @('data/', 'runs/', 'training_set/')) {
    $count = @($tracked | Where-Object { $_.StartsWith($prefix) }).Count
    Write-Output ("{0,-16} {1,8} files" -f $prefix, $count)
}
$trackedCaches = @($tracked | Where-Object { $_ -match '(^|/)__pycache__/' }).Count
Write-Output ("{0,-16} {1,8} files" -f '__pycache__/', $trackedCaches)

Write-Output ''
Write-Output '== Untracked source candidates after ignore rules =='
foreach ($path in @(
    'PROJECT_STRUCTURE.md',
    'REMOTE_SYNC.md',
    'configs',
    'dual_context_water',
    'training',
    'waterseg_platform',
    'src',
    'scripts',
    'tests',
    'docs'
)) {
    $count = @(git ls-files --others --exclude-standard -- $path).Count
    Write-Output ("{0,-28} {1,8} files" -f $path, $count)
}

Write-Output ''
Write-Output '== LFS =='
git lfs version
git lfs ls-files
$requiredModels = @(
    'onnx/floodnet_binary_aug_yolov8m_1024.onnx',
    'onnx/waterlogging_yolov8m_base_640.onnx',
    'onnx/waterlogging_yolov8m_hard_finetune_640.onnx',
    'onnx/dual_context_water_v2_large_1024_candidate.onnx',
    'platform_prediction_yolo_dotnet_export/onnx/floodnet_binary_aug_yolov8m_1024.onnx'
)
foreach ($modelPath in $requiredModels) {
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
        throw "Required deploy model is missing: $modelPath"
    }
    $filter = git check-attr filter -- $modelPath
    Write-Output $filter
    if ($filter -notmatch 'filter: lfs') {
        throw "Required ONNX is not covered by Git LFS: $modelPath"
    }
}

Write-Output ''
Write-Output '== Staged files larger than 100 MiB =='
$largeStaged = @()
foreach ($path in @(git diff --cached --name-only --diff-filter=ACM)) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        if ($item.Length -gt 100MB) {
            $largeStaged += [PSCustomObject]@{
                MiB  = [math]::Round($item.Length / 1MB, 1)
                Path = $path
            }
        }
    }
}
if ($largeStaged.Count -eq 0) {
    Write-Output 'None.'
} else {
    $largeStaged | Sort-Object MiB -Descending | Format-Table -AutoSize
    Write-Warning 'Confirm every large staged file is represented by an LFS pointer.'
}

Write-Output ''
Write-Output 'Audit complete. No files were modified, staged, committed, or pushed.'
