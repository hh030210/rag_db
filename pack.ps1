# pack.ps1 —— 打包要部署的文件到 dist/prompt_iteration_optimizer_<时间戳>.zip
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
if (-not $ROOT) { $ROOT = (Get-Location).Path }

$OUT_DIR = Join-Path $ROOT "dist"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$ZIP_FILE = Join-Path $OUT_DIR ("prompt_iteration_optimizer_" + $TS + ".zip")

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

$TMP = New-Item -ItemType Directory -Force -Path (Join-Path $env:TEMP ("pack_" + [Guid]::NewGuid()))
$TMP_CORE = New-Item -ItemType Directory -Force -Path (Join-Path $TMP.FullName "core")

foreach ($f in $CORE_FILES) {
    $src = Join-Path $ROOT $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $TMP_CORE.FullName
    } else {
        Write-Host ("[warn] " + $f + " missing, skipped")
    }
}

if (Test-Path $TRAIN_DIR) {
    $dst = Join-Path $TMP_CORE.FullName "code1_train_data"
    Copy-Item -Path $TRAIN_DIR -Destination $dst -Recurse
    Write-Host ("[info] training artifacts copied")
} else {
    Write-Host ("[warn] training dir " + $TRAIN_DIR + " missing")
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($TMP_CORE.FullName, $ZIP_FILE)

Remove-Item -Recurse -Force $TMP.FullName

Write-Host ("[done] " + $ZIP_FILE)
Write-Host "Contents:"
$archive = [System.IO.Compression.ZipFile]::OpenRead($ZIP_FILE)
$archive.Entries | ForEach-Object { Write-Host ("  " + $_.FullName) }
$archive.Dispose()
