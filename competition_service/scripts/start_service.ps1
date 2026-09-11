[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$BundleRoot,
    [Parameter(Mandatory)]
    [string]$BindHost,
    [int]$Port = 8000,
    [string]$StateFile,
    [switch]$ReplaceStaleState,
    [string]$AudioProxyUrl,
    [string[]]$AudioAllowedHost,
    [string]$AudioTempRoot,
    [double]$AudioBudgetSeconds = 20,
    [int]$MaxConcurrentAudio = 4
)

$ErrorActionPreference = "Stop"

function Resolve-BundleRoot {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "BundleRoot must be an existing directory."
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

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

function Assert-AudioConfiguration {
    param([string]$ProxyUrl, [string[]]$AllowedHosts)
    $hasProxy = -not [string]::IsNullOrWhiteSpace($ProxyUrl)
    $hasHosts = $null -ne $AllowedHosts -and $AllowedHosts.Count -gt 0
    if ($hasProxy -and -not $hasHosts) {
        throw "AudioProxyUrl requires at least one AudioAllowedHost."
    }
    if (-not $hasProxy -and $hasHosts) {
        throw "AudioAllowedHost requires AudioProxyUrl."
    }
    if (-not $hasProxy) { return }

    $proxyUri = $null
    if (-not [System.Uri]::TryCreate($ProxyUrl, [System.UriKind]::Absolute, [ref]$proxyUri) -or
        $proxyUri.Scheme -notin @("http", "https") -or
        -not [string]::IsNullOrEmpty($proxyUri.UserInfo) -or
        -not [string]::IsNullOrEmpty($proxyUri.Query) -or
        -not [string]::IsNullOrEmpty($proxyUri.Fragment) -or
        $proxyUri.AbsolutePath -notin @("", "/")) {
        throw "AudioProxyUrl must be a credential-free http or https origin."
    }
    foreach ($host in $AllowedHosts) {
        if ([string]::IsNullOrWhiteSpace($host) -or $host -ne $host.Trim() -or
            $host -match "[/:\\?#@]" -or [System.Uri]::CheckHostName($host) -ne [System.UriHostNameType]::Dns) {
            throw "AudioAllowedHost must contain DNS host names only."
        }
    }
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Value)

    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount += 1
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append((('\' * (($backslashCount * 2) + 1)) -join ''))
            [void]$builder.Append('"')
        }
        else {
            if ($backslashCount -gt 0) {
                [void]$builder.Append((('\' * $backslashCount) -join ''))
            }
            [void]$builder.Append($character)
        }
        $backslashCount = 0
    }
    if ($backslashCount -gt 0) {
        [void]$builder.Append((('\' * ($backslashCount * 2)) -join ''))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Get-NormalizedProcessCreationTime {
    param($Process)

    if ($null -eq $Process -or $null -eq $Process.CreationDate) {
        return $null
    }
    try {
        return ([DateTime]$Process.CreationDate).ToUniversalTime().ToString("o")
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

function Acquire-StateReservation {
    param([string]$StatePath)

    $reservationPath = "$($StatePath).launch.lock"
    try {
        return [System.IO.File]::Open(
            $reservationPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        throw "Another launcher currently owns the state reservation. Retry after it finishes."
    }
}

function Get-StateSnapshot {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $file = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    $content = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = ([BitConverter]::ToString($hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($content)))).Replace("-", "")
    }
    finally {
        $hasher.Dispose()
    }
    return [pscustomobject]@{
        content = $content
        length = $file.Length
        creation_time_utc = $file.CreationTimeUtc.Ticks
        last_write_time_utc = $file.LastWriteTimeUtc.Ticks
        sha256 = $hash
    }
}

function Test-StateSnapshotUnchanged {
    param([string]$Path, $Snapshot)

    if ($null -eq $Snapshot) {
        return -not (Test-Path -LiteralPath $Path -PathType Leaf)
    }
    $current = Get-StateSnapshot -Path $Path
    return $null -ne $current -and
        $current.length -eq $Snapshot.length -and
        $current.creation_time_utc -eq $Snapshot.creation_time_utc -and
        $current.last_write_time_utc -eq $Snapshot.last_write_time_utc -and
        $current.sha256 -ceq $Snapshot.sha256 -and
        $current.content -ceq $Snapshot.content
}

function Get-ExistingStateStatus {
    param($Snapshot)

    if ($null -eq $Snapshot) {
        return "Absent"
    }
    try {
        $state = $Snapshot.content | ConvertFrom-Json
        $expectedKeys = @("pid", "bind_host", "port", "start_time_utc", "bundle_root")
        $actualKeys = @($state.PSObject.Properties.Name | Sort-Object)
        if (($actualKeys -join ",") -ne (($expectedKeys | Sort-Object) -join ",") -or
            $state.pid -isnot [long] -or $state.port -isnot [long] -or
            $state.bind_host -isnot [string] -or $state.bundle_root -isnot [string] -or
            $state.start_time_utc -isnot [string] -or $state.pid -lt 1 -or
            $state.port -lt 1 -or $state.port -gt 65535) {
            return "StaleOrInvalid"
        }
        $normalizedStateTime = ([DateTime]::Parse(
            $state.start_time_utc,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind
        )).ToUniversalTime().ToString("o")
        if ($normalizedStateTime -cne $state.start_time_utc) {
            return "StaleOrInvalid"
        }
        $stateRoot = (Resolve-Path -LiteralPath $state.bundle_root).Path
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$state.pid)"
        if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine) -or
            $process.CommandLine -notmatch "(?i)competition_emotion\.service") {
            return "StaleOrInvalid"
        }
        $commandBundleRoots = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--bundle-root")
        $commandHosts = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--host")
        $commandPorts = @(Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--port")
        if ($commandBundleRoots.Count -ne 1 -or $commandHosts.Count -ne 1 -or $commandPorts.Count -ne 1 -or
            (Resolve-Path -LiteralPath $commandBundleRoots[0]).Path -cne $stateRoot -or
            $commandHosts[0] -cne $state.bind_host -or $commandPorts[0] -cne ([string]$state.port) -or
            (Get-NormalizedProcessCreationTime -Process $process) -cne $state.start_time_utc) {
            return "StaleOrInvalid"
        }
        return "LiveOwned"
    }
    catch {
        return "StaleOrInvalid"
    }
}

$startedProcess = $null
$resolvedStateFile = $null
$stateReservation = $null
$statePublishedByThisInvocation = $false
try {
    if (-not ($Port -is [int]) -or $Port -lt 1 -or $Port -gt 65535) {
        throw "Port must be an integer between 1 and 65535."
    }
    if ([double]::IsNaN($AudioBudgetSeconds) -or [double]::IsInfinity($AudioBudgetSeconds) -or $AudioBudgetSeconds -le 0 -or $AudioBudgetSeconds -gt 25) {
        throw "AudioBudgetSeconds must be greater than 0 and no greater than 25."
    }
    if (-not ($MaxConcurrentAudio -is [int]) -or $MaxConcurrentAudio -lt 1) {
        throw "MaxConcurrentAudio must be a positive integer."
    }

    $canonicalBindHost = Get-CanonicalBindHost -HostValue $BindHost
    Assert-AudioConfiguration -ProxyUrl $AudioProxyUrl -AllowedHosts $AudioAllowedHost
    $resolvedBundleRoot = Resolve-BundleRoot -Path $BundleRoot
    $currentPath = Join-Path $resolvedBundleRoot "current.json"
    if (-not (Test-Path -LiteralPath $currentPath -PathType Leaf)) {
        throw "BundleRoot does not contain current.json."
    }
    if ([string]::IsNullOrWhiteSpace($StateFile)) {
        $resolvedStateFile = Join-Path $resolvedBundleRoot "service-state.json"
    }
    else {
        $resolvedStateFile = [System.IO.Path]::GetFullPath($StateFile)
    }
    $stateDirectory = Split-Path -Parent $resolvedStateFile
    if ([string]::IsNullOrWhiteSpace($stateDirectory)) {
        throw "StateFile must include a directory."
    }
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    $stateReservation = Acquire-StateReservation -StatePath $resolvedStateFile
    $existingStateSnapshot = Get-StateSnapshot -Path $resolvedStateFile
    $existingStateStatus = Get-ExistingStateStatus -Snapshot $existingStateSnapshot
    if ($existingStateStatus -eq "LiveOwned") {
        throw "A live owned service state already exists. Refusing to overwrite it."
    }
    if ($existingStateStatus -eq "StaleOrInvalid") {
        if (-not $ReplaceStaleState) {
            throw "State file is stale or invalid. Re-run with -ReplaceStaleState after operator review."
        }
        if (-not (Test-StateSnapshotUnchanged -Path $resolvedStateFile -Snapshot $existingStateSnapshot)) {
            throw "State file changed during stale-state review. Refusing to remove it."
        }
        Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop
    }

    $logDirectory = Join-Path $resolvedBundleRoot "logs\\service"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $stdoutPath = Join-Path $logDirectory "service-$timestamp.stdout.log"
    $stderrPath = Join-Path $logDirectory "service-$timestamp.stderr.log"
    $serviceRoot = Split-Path -Parent $PSScriptRoot
    $arguments = @(
        "-3.12", "-m", "competition_emotion.service",
        "--bundle-root", $resolvedBundleRoot,
        "--host", $canonicalBindHost,
        "--port", $Port,
        "--audio-budget-seconds", $AudioBudgetSeconds,
        "--max-concurrent-audio", $MaxConcurrentAudio
    )
    if ($hasProxy = -not [string]::IsNullOrWhiteSpace($AudioProxyUrl)) {
        $arguments += "--audio-proxy-url", $AudioProxyUrl
        foreach ($host in $AudioAllowedHost) {
            $arguments += "--audio-allowed-host", $host
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($AudioTempRoot)) {
        $arguments += "--audio-temp-root", ([System.IO.Path]::GetFullPath($AudioTempRoot))
    }
    $argumentLine = (@($arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument -Value ([string]$_) }) -join " ")

    $startedProcess = Start-Process -FilePath "py" -ArgumentList $argumentLine -WorkingDirectory $serviceRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Start-Sleep -Milliseconds 150
    $verifiedProcess = Get-Process -Id $startedProcess.Id -ErrorAction Stop
    if ($null -eq $verifiedProcess -or $verifiedProcess.HasExited) {
        throw "Service process exited before state publication. Inspect the service logs."
    }
    $processMetadata = Get-CimInstance Win32_Process -Filter "ProcessId = $startedProcess.Id"
    $processCreationTimeUtc = Get-NormalizedProcessCreationTime -Process $processMetadata
    if ([string]::IsNullOrWhiteSpace($processCreationTimeUtc)) {
        throw "Service process creation time could not be verified before state publication."
    }

    $state = [ordered]@{
        pid = $startedProcess.Id
        bind_host = $canonicalBindHost
        port = $Port
        start_time_utc = $processCreationTimeUtc
        bundle_root = $resolvedBundleRoot
    }
    $temporaryStatePath = Join-Path $stateDirectory ("." + [System.IO.Path]::GetFileName($resolvedStateFile) + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    try {
        [System.IO.File]::WriteAllText($temporaryStatePath, ($state | ConvertTo-Json -Compress), [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryStatePath -Destination $resolvedStateFile -Force
        $statePublishedByThisInvocation = $true
    }
    finally {
        Remove-Item -LiteralPath $temporaryStatePath -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Service started: PID $($startedProcess.Id), $canonicalBindHost`:$Port"
    Write-Host "State file: $resolvedStateFile"
    Write-Host "Logs: $logDirectory"
}
catch {
    if ($null -ne $startedProcess) {
        Stop-Process -InputObject $startedProcess -Force -ErrorAction SilentlyContinue
    }
    if ($statePublishedByThisInvocation) {
        Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction SilentlyContinue
    }
    throw
}
finally {
    if ($null -ne $stateReservation) {
        $stateReservation.Dispose()
    }
}
