[CmdletBinding()]
param(
    [string]$Workbook = "F:\netease\_music\competition\data\emotion_songs_20260910.xlsx",
    [string]$BundleRoot = "F:\netease\_music\competition\runs\official-20260910"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Workbook -PathType Leaf)) {
    throw "Official workbook was not found: $Workbook"
}

$serviceRoot = Split-Path -Parent $PSScriptRoot
Push-Location $serviceRoot
try {
    & py -3.12 -m competition_emotion.train --workbook $Workbook --bundle-dir $BundleRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Official training exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$currentPath = Join-Path $BundleRoot "current.json"
if (-not (Test-Path -LiteralPath $currentPath -PathType Leaf)) {
    throw "Training did not publish current.json: $currentPath"
}

$current = Get-Content -LiteralPath $currentPath -Raw | ConvertFrom-Json
$activeBundle = [string]$current.active_bundle
if ([string]::IsNullOrWhiteSpace($activeBundle)) {
    throw "current.json has no active_bundle"
}

$reportPath = Join-Path $BundleRoot (($activeBundle -replace "/", "\") + "\report.json")
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    throw "Training did not publish the active report: $reportPath"
}

Write-Host "Active bundle: $(Split-Path -Parent $reportPath)"
Write-Host "Active report: $reportPath"
