<#
pack_upload.ps1 —— Windows 端把要上传到 Linux 服务器的文件打成 zip
用法（在 PowerShell 中）：
    powershell -ExecutionPolicy Bypass -File pack_upload.ps1
产出：
    .\dist\prompt_iteration_optimizer_<时间戳>.zip
#>
$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
if (-not $ROOT) { $ROOT = (Get-Location).Path }

$OUT_DIR = Join-Path $ROOT "dist"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$ZIP_FILE = Join-Path $OUT_DIR "prompt_iteration_optimizer_${TS}.zip"

$CORE_FILES = @(
    "prompt_iteration_optimizer.py",
    "prompt_iteration_service.py",
    "case_level_service.py",
    "case_level_optimizer_service.py",
    "api_server.py",
    "requirements_api.txt",
    "requirements_bge.txt",
    "requirements_numpy.txt",
    "setup.sh",
    "start.sh",
    "smoke_test.sh",
    "README_api.md",
    "DEPLOY.md"
)

$TRAIN_DIR = Join-Path $ROOT "code1\chapter3_backup\codes\bylw_rag\new_experiments"

$TMP = New-Item -ItemType Directory -Force -Path (Join-Path $env:TEMP ("pack_" + [Guid]::NewGuid())) | Out-Null
$TMP_CORE = New-Item -ItemType Directory -Force -Path (Join-Path $TMP.FullName "core") | Out-Null

foreach ($f in $CORE_FILES) {
    $src = Join-Path $ROOT $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $TMP_CORE.FullName
    } else {
        Write-Host "[警告] $f 不存在，跳过" -ForegroundColor Yellow
    }
}

if (Test-Path $TRAIN_DIR) {
    $dst = Join-Path $TMP_CORE.FullName "code1_train_data"
    Copy-Item -Path $TRAIN_DIR -Destination $dst -Recurse
    Write-Host "[信息] 已包含训练产物: $TRAIN_DIR" -ForegroundColor Green
} else {
    Write-Host "[警告] 训练产物目录 $TRAIN_DIR 不存在，" -ForegroundColor Yellow
    Write-Host "       上传后请把对应文件放到 ./code1/chapter3_backup/... 路径下。" -ForegroundColor Yellow
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($TMP_CORE.FullName, $ZIP_FILE)

Remove-Item -Recurse -Force $TMP.FullName

Write-Host "[完成] $ZIP_FILE" -ForegroundColor Green
Write-Host ""
Write-Host "打包内容："
$archive = [System.IO.Compression.ZipFile]::OpenRead($ZIP_FILE)
$archive.Entries | ForEach-Object { Write-Host ("  " + $_.FullName) }
$archive.Dispose()