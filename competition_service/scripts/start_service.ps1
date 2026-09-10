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

function Resolve-BundleRoot {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "BundleRoot must be an existing directory."
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-BindHost {
    param([string]$HostValue)
    $address = [System.Net.IPAddress]::None
    if (-not [System.Net.IPAddress]::TryParse($HostValue, [ref]$address)) {
        throw "BindHost must be an IP literal; DNS names are not accepted."
    }
    if ([System.Net.IPAddress]::IsLoopback($address)) {
        throw "BindHost must not be a loopback address."
    }
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

$startedProcess = $null
$resolvedStateFile = $null
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

    Assert-BindHost -HostValue $BindHost
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

    $logDirectory = Join-Path $resolvedBundleRoot "logs\\service"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $stdoutPath = Join-Path $logDirectory "service-$timestamp.stdout.log"
    $stderrPath = Join-Path $logDirectory "service-$timestamp.stderr.log"
    $serviceRoot = Split-Path -Parent $PSScriptRoot
    $arguments = @(
        "-3.12", "-m", "competition_emotion.service",
        "--bundle-root", $resolvedBundleRoot,
        "--host", $BindHost,
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

    $state = [ordered]@{
        pid = $startedProcess.Id
        bind_host = $BindHost
        port = $Port
        start_time_utc = [DateTime]::UtcNow.ToString("o")
        bundle_root = $resolvedBundleRoot
    }
    $temporaryStatePath = Join-Path $stateDirectory ("." + [System.IO.Path]::GetFileName($resolvedStateFile) + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    try {
        [System.IO.File]::WriteAllText($temporaryStatePath, ($state | ConvertTo-Json -Compress), [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryStatePath -Destination $resolvedStateFile -Force
    }
    finally {
        Remove-Item -LiteralPath $temporaryStatePath -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Service started: PID $($startedProcess.Id), $BindHost`:$Port"
    Write-Host "State file: $resolvedStateFile"
    Write-Host "Logs: $logDirectory"
}
catch {
    if ($null -ne $startedProcess) {
        Stop-Process -Id $startedProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $resolvedStateFile) {
        Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction SilentlyContinue
    }
    throw
}
