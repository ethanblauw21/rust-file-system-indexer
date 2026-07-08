<#
.SYNOPSIS
    Optional convenience script for file_indexer's model setup (see README.md "Model Setup").

.DESCRIPTION
    file_indexer needs three things for dense/hybrid search:
      1. nomic-embed-text-v1.5.onnx  (Hugging Face: nomic-ai/nomic-embed-text-v1.5-ONNX)
      2. tokenizer.json              (same Hugging Face repo)
      3. an onnxruntime 1.24.x shared library (onnxruntime.dll on Windows)

    This script does NOT download any of the three files for you — the exact Hugging Face
    file paths and the correct onnxruntime release asset for your OS/arch are things this
    script's author has not verified against a live fetch, and guessing a URL here would be
    exactly the kind of unverified claim this project's doc-truth audit exists to avoid making.
    Get files 1 and 2 from:
        https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-ONNX
    Get file 3 (pick the 1.24.x asset matching your OS/arch) from:
        https://github.com/microsoft/onnxruntime/releases

    What this script DOES do:
      - Creates the destination directory for files 1+2.
      - After you've placed nomic-embed-text-v1.5.onnx and tokenizer.json in it, verifies both
        expected filenames are present (src/indexer.rs:408-409 requires these exact names).
      - Prints the exact file_indexer.toml / environment-variable lines to point
        file_indexer at your downloads, per the resolution order documented in
        CLAUDE.md's "Configuration file" design-decision bullet.

.PARAMETER ModelDir
    Directory that will hold (or already holds) nomic-embed-text-v1.5.onnx and tokenizer.json.
    Defaults to a "models\nomic" folder next to this script's repo checkout.

.PARAMETER OrtDylibPath
    Full path to the onnxruntime shared library (e.g. ...\lib\onnxruntime.dll), if you already
    know it. Optional — omit and fill in the printed placeholder later.

.EXAMPLE
    .\docs\setup_model.ps1 -ModelDir C:\models\nomic -OrtDylibPath C:\onnxruntime\lib\onnxruntime.dll
#>
param(
    [string]$ModelDir = (Join-Path $PSScriptRoot "..\models\nomic"),
    [string]$OrtDylibPath = ""
)

$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)

if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    Write-Host "Created $ModelDir"
} else {
    Write-Host "Using existing directory $ModelDir"
}

$modelFile = Join-Path $ModelDir "nomic-embed-text-v1.5.onnx"
$tokenizerFile = Join-Path $ModelDir "tokenizer.json"

Write-Host ""
Write-Host "Required files (exact names — src/indexer.rs:408-409):"
if (Test-Path $modelFile) {
    Write-Host "  [ok]      $modelFile"
} else {
    Write-Host "  [missing] $modelFile"
    Write-Host "            Download from https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-ONNX"
}
if (Test-Path $tokenizerFile) {
    Write-Host "  [ok]      $tokenizerFile"
} else {
    Write-Host "  [missing] $tokenizerFile"
    Write-Host "            Download from https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-ONNX"
}

Write-Host ""
if (-not $OrtDylibPath) {
    Write-Host "ONNX Runtime shared library: not provided (-OrtDylibPath)."
    Write-Host "  Download onnxruntime 1.24.x for your OS/arch from"
    Write-Host "  https://github.com/microsoft/onnxruntime/releases and pass its path via -OrtDylibPath,"
    Write-Host "  or fill it in manually below."
    $OrtDylibPath = "<path to onnxruntime.dll / .so / .dylib>"
} else {
    Write-Host "ONNX Runtime shared library: $OrtDylibPath"
}

Write-Host ""
Write-Host "== Option A: file_indexer.toml (at <index-dir>/file_indexer.toml or ./file_indexer.toml) =="
Write-Host @"
[embedder]
onnx_model_dir = "$($ModelDir -replace '\\','/')"
ort_dylib_path = "$($OrtDylibPath -replace '\\','/')"
"@

Write-Host ""
Write-Host "== Option B: environment variables (used only as a fallback when the config file doesn't set these) =="
Write-Host "  `$env:NOMIC_ONNX_PATH = `"$ModelDir`""
Write-Host "  `$env:ORT_DYLIB_PATH  = `"$OrtDylibPath`""

Write-Host ""
Write-Host "Without either set, 'index' hard-errors unless you pass --no-embed (sparse-only index)."
