[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BundleRoot,
    [Parameter(Mandatory)]
    [string]$BindHost,
    [int]$Port = 8000,
    [string]$StateFile,
    [string]$AudioProxyUrl,
    [string[]]$AudioAllowedHost,
    [string]$AudioTempRoot,
    [double]$AudioBudgetSeconds = 20,
    [int]$MaxConcurrentAudio = 4
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $BundleRoot -PathType Container)) {
    throw "BundleRoot must be an existing directory."
}
$resolvedBundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
if ([string]::IsNullOrWhiteSpace($StateFile)) {
    $resolvedStateFile = Join-Path $resolvedBundleRoot "service-state.json"
}
else {
    $resolvedStateFile = [System.IO.Path]::GetFullPath($StateFile)
}
if (-not (Test-Path -LiteralPath $resolvedStateFile -PathType Leaf)) {
    throw "State file was not found. Refusing to stop any process."
}
try {
    $state = Get-Content -LiteralPath $resolvedStateFile -Raw | ConvertFrom-Json
}
catch {
    throw "Invalid state schema. Refusing to stop any process."
}
$expectedKeys = @("pid", "bind_host", "port", "start_time_utc", "bundle_root")
$actualKeys = @($state.PSObject.Properties.Name | Sort-Object)
if (($actualKeys -join ",") -ne (($expectedKeys | Sort-Object) -join ",") -or
    $state.pid -isnot [long] -or $state.port -isnot [long] -or
    $state.bind_host -isnot [string] -or $state.bundle_root -isnot [string] -or
    $state.start_time_utc -isnot [string] -or $state.pid -lt 1 -or $state.port -lt 1 -or $state.port -gt 65535) {
    throw "Invalid state schema. Refusing to stop any process."
}
try {
    $stateBundleRoot = (Resolve-Path -LiteralPath $state.bundle_root).Path
}
catch {
    throw "Invalid state schema. Refusing to stop any process."
}
if ($stateBundleRoot -ne $resolvedBundleRoot -or $state.bind_host -ne $BindHost -or $state.port -ne $Port) {
    throw "State does not match the requested nonsecret service settings. Refusing to stop any process."
}

$statePid = [int]$state.pid
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $statePid"
if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine) -or
    $process.CommandLine -notmatch "(?i)competition_emotion\.service") {
    throw "State PID is not the expected service process. Refusing to stop any process."
}
$bundleMatch = [regex]::Match($process.CommandLine, '(?i)(?:^|\s)--bundle-root\s+(?:"(?<bundle>[^"]+)"|(?<bundle>\S+))')
if (-not $bundleMatch.Success) {
    throw "State PID has no readable bundle root. Refusing to stop any process."
}
$commandBundleRoot = $bundleMatch.Groups["bundle"].Value
try {
    $resolvedCommandBundleRoot = (Resolve-Path -LiteralPath $commandBundleRoot).Path
}
catch {
    throw "State PID bundle root cannot be verified. Refusing to stop any process."
}
if ($resolvedCommandBundleRoot -ne $resolvedBundleRoot) {
    throw "State PID does not own the requested bundle root. Refusing to stop any process."
}

Stop-Process -Id $statePid -Force
Wait-Process -Id $statePid -Timeout 15 -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop

$startScript = Join-Path $PSScriptRoot "start_service.ps1"
& $startScript -BundleRoot $resolvedBundleRoot -BindHost $BindHost -Port $Port -StateFile $resolvedStateFile `
    -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost -AudioTempRoot $AudioTempRoot `
    -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio
