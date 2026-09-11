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
    [int]$MaxConcurrentAudio = 4,
    [string]$RubricPath,
    [string]$TracePath
)

$ErrorActionPreference = "Stop"

function Get-CanonicalBindHost {
    param([string]$HostValue)

    $address = [System.Net.IPAddress]::None
    if (-not [System.Net.IPAddress]::TryParse($HostValue, [ref]$address)) {
        throw "BindHost must be an IP literal; DNS names are not accepted."
    }
    if ([System.Net.IPAddress]::IsLoopback($address)) {
        throw "BindHost must not be a loopback address."
    }
    return $address.ToString()
}

function Get-NormalizedProcessCreationTime {
    param($Process)

    if ($null -eq $Process -or $null -eq $Process.CreationDate) {
        return $null
    }
    try {
        return ([DateTime]$Process.CreationDate).ToUniversalTime().ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    }
    catch {
        return $null
    }
}

function Get-CommandLineOptionValues {
    param([string]$CommandLine, [string]$Option)

    $pattern = '(?i)(?:^|\s)"?' + [regex]::Escape($Option) + '"?\s+(?:"(?<value>[^"]+)"|(?<value>\S+))'
    return @([regex]::Matches($CommandLine, $pattern) | ForEach-Object { $_.Groups["value"].Value })
}

$startScript = Join-Path $PSScriptRoot "start_service.ps1"
. $startScript -BundleRoot $BundleRoot -BindHost $BindHost -Port $Port -StateFile $StateFile `
    -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost -AudioTempRoot $AudioTempRoot `
    -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio `
    -RubricPath $RubricPath -TracePath $TracePath

$configuration = Get-ServiceLaunchConfiguration -BundleRoot $BundleRoot -BindHost $BindHost -Port $Port `
    -StateFile $StateFile -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost `
    -AudioTempRoot $AudioTempRoot -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio `
    -RubricPath $RubricPath -TracePath $TracePath
$resolvedBundleRoot = $configuration.resolved_bundle_root
$canonicalBindHost = $configuration.canonical_bind_host
$resolvedStateFile = $configuration.resolved_state_file
$stateReservation = $null
try {
    $stateReservation = Acquire-StateReservation -StatePath $resolvedStateFile
if (-not (Test-Path -LiteralPath $resolvedStateFile -PathType Leaf)) {
    throw "State file was not found. Refusing to stop any process."
}
try {
    $state = ConvertFrom-ServiceStateJson -Content (Get-Content -LiteralPath $resolvedStateFile -Raw)
}
catch {
    throw "Invalid state schema. Refusing to stop any process."
}
$expectedKeys = @("pid", "bind_host", "port", "start_time_utc", "bundle_root")
$actualKeys = @($state.PSObject.Properties.Name | Sort-Object)
if (($actualKeys -join ",") -ne (($expectedKeys | Sort-Object) -join ",") -or
    (($state.pid -isnot [int]) -and ($state.pid -isnot [long])) -or
    (($state.port -isnot [int]) -and ($state.port -isnot [long])) -or
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
if ($stateBundleRoot -cne $resolvedBundleRoot -or $state.bind_host -cne $canonicalBindHost -or $state.port -ne $Port) {
    throw "State does not match the requested nonsecret service settings. Refusing to stop any process."
}

$statePid = [int]$state.pid
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $statePid"
if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine) -or
    $process.CommandLine -notmatch "(?i)competition_emotion\.service") {
    throw "State PID is not the expected service process. Refusing to stop any process."
}
$commandBundleRoots = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--bundle-root")
$commandHosts = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--host")
$commandPorts = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--port")
if ($commandBundleRoots.Count -ne 1 -or $commandHosts.Count -ne 1 -or $commandPorts.Count -ne 1) {
    throw "State PID command line is ambiguous. Refusing to stop any process."
}
try {
    $resolvedCommandBundleRoot = (Resolve-Path -LiteralPath $commandBundleRoots[0]).Path
}
catch {
    throw "State PID bundle root cannot be verified. Refusing to stop any process."
}
if ($resolvedCommandBundleRoot -cne $resolvedBundleRoot -or
    $commandHosts[0] -cne $state.bind_host -or $commandPorts[0] -cne ([string]$state.port)) {
    throw "State PID does not own the requested bundle root. Refusing to stop any process."
}
if ((Get-NormalizedProcessCreationTime -Process $process) -cne $state.start_time_utc) {
    throw "State PID process creation time does not match. Refusing to stop any process."
}

Stop-Process -Id $statePid -Force
Wait-Process -Id $statePid -Timeout 15 -ErrorAction SilentlyContinue
if ($null -ne (Get-Process -Id $statePid -ErrorAction SilentlyContinue)) {
    throw "State PID did not stop. Refusing to remove state."
}
Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop

Start-CompetitionEmotionService -Configuration $configuration -StateReservation $stateReservation -ReplaceStaleState `
    -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost -AudioTempRoot $AudioTempRoot `
    -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio `
    -RubricPath $RubricPath -TracePath $TracePath
}
finally {
    if ($null -ne $stateReservation) {
        $stateReservation.Dispose()
    }
}
