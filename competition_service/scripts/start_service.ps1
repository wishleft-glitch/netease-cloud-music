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
    [int]$MaxConcurrentAudio = 4,
    [string]$RubricPath,
    [string]$TracePath
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
        return ([DateTime]$Process.CreationDate).ToUniversalTime().ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    }
    catch {
        return $null
    }
}

function Get-CanonicalStateStartTimeUtc {
    param([string]$Content, $Value)

    $matches = @([regex]::Matches(
        $Content,
        '"start_time_utc"\s*:\s*"(?<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{7}Z)"'
    ))
    if ($matches.Count -ne 1) {
        throw "Invalid state schema. Refusing to use service state."
    }
    $rawTimestamp = $matches[0].Groups["timestamp"].Value
    $parsedTimestamp = [DateTime]::MinValue
    if (-not [DateTime]::TryParseExact(
        $rawTimestamp,
        "o",
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind,
        [ref]$parsedTimestamp
    ) -or $parsedTimestamp.Kind -ne [DateTimeKind]::Utc) {
        throw "Invalid state schema. Refusing to use service state."
    }
    $canonicalTimestamp = $parsedTimestamp.ToUniversalTime().ToString("o", [Globalization.CultureInfo]::InvariantCulture)
    if ($rawTimestamp -cne $canonicalTimestamp) {
        throw "Invalid state schema. Refusing to use service state."
    }
    if ($Value -is [DateTime]) {
        if ($Value.ToUniversalTime().ToString("o", [Globalization.CultureInfo]::InvariantCulture) -cne $canonicalTimestamp) {
            throw "Invalid state schema. Refusing to use service state."
        }
        return $canonicalTimestamp
    }
    if ($Value -is [string] -and $Value -ceq $canonicalTimestamp) {
        return $canonicalTimestamp
    }
    throw "Invalid state schema. Refusing to use service state."
}

function ConvertFrom-ServiceStateJson {
    param([string]$Content)

    try {
        $convertFromJson = Get-Command ConvertFrom-Json -ErrorAction Stop
        if ($convertFromJson.Parameters.ContainsKey("DateKind")) {
            $state = $Content | ConvertFrom-Json -DateKind String
        }
        else {
            $state = $Content | ConvertFrom-Json
        }
        if ($null -eq $state) {
            throw "State content is empty."
        }
        $state.start_time_utc = Get-CanonicalStateStartTimeUtc -Content $Content -Value $state.start_time_utc
        return $state
    }
    catch {
        throw "Invalid state schema. Refusing to use service state."
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

function Get-StaleStateBackupPath {
    param(
        [string]$StatePath,
        [string]$StateDirectory
    )

    $fileName = [System.IO.Path]::GetFileName($StatePath)
    do {
        $candidate = Join-Path $StateDirectory ("." + $fileName + "." + [Guid]::NewGuid().ToString("N") + ".stale")
    } while (Test-Path -LiteralPath $candidate)
    return $candidate
}

function Remove-PublishedStateIfOwned {
    param(
        [string]$resolvedStateFile,
        $PublishedSnapshot
    )

    try {
        if ($null -eq $PublishedSnapshot -or
            -not (Test-StateSnapshotUnchanged -Path $resolvedStateFile -Snapshot $PublishedSnapshot)) {
            return $false
        }
        Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Test-StaleStateBackupOwned {
    param(
        [string]$BackupPath,
        $BackupSnapshot
    )

    try {
        return $null -ne $BackupSnapshot -and
            (Test-StateSnapshotUnchanged -Path $BackupPath -Snapshot $BackupSnapshot)
    }
    catch {
        return $false
    }
}

function Get-ExistingStateStatus {
    param($Snapshot)

    if ($null -eq $Snapshot) {
        return "Absent"
    }
    try {
        $state = ConvertFrom-ServiceStateJson -Content $Snapshot.content
        $expectedKeys = @("pid", "bind_host", "port", "start_time_utc", "bundle_root")
        $actualKeys = @($state.PSObject.Properties.Name | Sort-Object)
        if (($actualKeys -join ",") -ne (($expectedKeys | Sort-Object) -join ",") -or
            (($state.pid -isnot [int]) -and ($state.pid -isnot [long])) -or
            (($state.port -isnot [int]) -and ($state.port -isnot [long])) -or
            $state.bind_host -isnot [string] -or $state.bundle_root -isnot [string] -or
            $state.start_time_utc -isnot [string] -or $state.pid -lt 1 -or
            $state.port -lt 1 -or $state.port -gt 65535) {
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

function Get-ServiceLaunchConfiguration {
    param(
        [string]$BundleRoot,
        [string]$BindHost,
        [int]$Port,
        [string]$StateFile,
        [string]$AudioProxyUrl,
        [string[]]$AudioAllowedHost,
        [string]$AudioTempRoot,
        [double]$AudioBudgetSeconds,
        [int]$MaxConcurrentAudio,
        [string]$RubricPath,
        [string]$TracePath
    )

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
    return [pscustomobject]@{
        canonical_bind_host = $canonicalBindHost
        resolved_bundle_root = $resolvedBundleRoot
        resolved_state_file = $resolvedStateFile
        state_directory = $stateDirectory
        port = $Port
        rubric_path = $RubricPath
        trace_path = $TracePath
    }
}

function Start-CompetitionEmotionService {
    param(
        [Parameter(Mandatory)]$Configuration,
        [Parameter(Mandatory)][System.IO.FileStream]$StateReservation,
        [switch]$ReplaceStaleState,
        [string]$AudioProxyUrl,
        [string[]]$AudioAllowedHost,
        [string]$AudioTempRoot,
        [double]$AudioBudgetSeconds,
        [int]$MaxConcurrentAudio,
        [string]$RubricPath,
        [string]$TracePath
    )

    $allowedCallerPaths = @(
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "start_service.ps1")),
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "restart_service.ps1"))
    )
    $callerPath = if ([string]::IsNullOrWhiteSpace($MyInvocation.ScriptName)) {
        $null
    }
    else {
        [System.IO.Path]::GetFullPath($MyInvocation.ScriptName)
    }
    $expectedReservationPath = "$($Configuration.resolved_state_file).launch.lock"
    if ($null -eq $callerPath -or $callerPath -notin $allowedCallerPaths -or
        -not $StateReservation.CanRead -or -not $StateReservation.CanWrite -or
        $StateReservation.Name -cne $expectedReservationPath) {
        throw "The service launch helper requires a matching reservation from the start or restart script."
    }

    $startedProcess = $null
    $statePublishedByThisInvocation = $false
    $publishedStateSnapshot = $null
    $staleStateBackupPath = $null
    $staleStateBackupSnapshot = $null
    $staleStateBackupOwnedByInvocation = $false
    $resolvedStateFile = $null
    try {
        $resolvedStateFile = $Configuration.resolved_state_file
        $resolvedBundleRoot = $Configuration.resolved_bundle_root
        $canonicalBindHost = $Configuration.canonical_bind_host
        $stateDirectory = $Configuration.state_directory
        $Port = [int]$Configuration.port
        $RubricPath = $Configuration.rubric_path
        $TracePath = $Configuration.trace_path
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
            $staleStateBackupPath = Get-StaleStateBackupPath -StatePath $resolvedStateFile -StateDirectory $stateDirectory
            Move-Item -LiteralPath $resolvedStateFile -Destination $staleStateBackupPath -ErrorAction Stop
            $staleStateBackupOwnedByInvocation = $true
            $staleStateBackupSnapshot = Get-StateSnapshot -Path $staleStateBackupPath
            if ($null -eq $staleStateBackupSnapshot) {
                throw "Stale state backup could not be verified before launch."
            }
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
        if (-not [string]::IsNullOrWhiteSpace($RubricPath)) {
            $arguments += "--rubric-path", ([System.IO.Path]::GetFullPath($RubricPath))
        }
        if (-not [string]::IsNullOrWhiteSpace($TracePath)) {
            $arguments += "--trace-path", ([System.IO.Path]::GetFullPath($TracePath))
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
            if (Test-Path -LiteralPath $resolvedStateFile) {
                throw "State file appeared during launch. Refusing to overwrite it."
            }
            Move-Item -LiteralPath $temporaryStatePath -Destination $resolvedStateFile -Force
            $statePublishedByThisInvocation = $true
            $publishedStateSnapshot = Get-StateSnapshot -Path $resolvedStateFile
            if ($null -eq $publishedStateSnapshot) {
                throw "Published service state could not be verified."
            }
            if ($staleStateBackupOwnedByInvocation) {
                if (-not (Test-StaleStateBackupOwned -BackupPath $staleStateBackupPath -BackupSnapshot $staleStateBackupSnapshot)) {
                    throw "Stale state backup changed during launch; retaining it for manual recovery."
                }
                Remove-Item -LiteralPath $staleStateBackupPath -Force -ErrorAction Stop
                $staleStateBackupOwnedByInvocation = $false
            }
        }
        finally {
            Remove-Item -LiteralPath $temporaryStatePath -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Service started: PID $($startedProcess.Id), $canonicalBindHost`:$Port"
        Write-Host "State file: $resolvedStateFile"
        Write-Host "Logs: $logDirectory"
    }
    catch {
        $launchError = $_
        if ($null -ne $startedProcess) {
            Stop-Process -InputObject $startedProcess -Force -ErrorAction SilentlyContinue
        }
        if ($statePublishedByThisInvocation) {
            [void](Remove-PublishedStateIfOwned -resolvedStateFile $resolvedStateFile -PublishedSnapshot $publishedStateSnapshot)
        }
        if ($staleStateBackupOwnedByInvocation -and
            (Test-Path -LiteralPath $staleStateBackupPath -PathType Leaf)) {
            if (-not (Test-StaleStateBackupOwned -BackupPath $staleStateBackupPath -BackupSnapshot $staleStateBackupSnapshot)) {
                Write-Warning "Stale state backup '$staleStateBackupPath' changed or lost ownership; it was retained for manual recovery."
            }
            elseif (-not (Test-Path -LiteralPath $resolvedStateFile)) {
                try {
                    Move-Item -LiteralPath $staleStateBackupPath -Destination $resolvedStateFile -ErrorAction Stop
                    $staleStateBackupOwnedByInvocation = $false
                }
                catch [System.Exception] {
                    Write-Warning "Could not restore stale state backup '$staleStateBackupPath'; it was retained for manual recovery."
                }
            }
            else {
                Write-Warning "A newer state exists at '$resolvedStateFile'; stale backup '$staleStateBackupPath' was retained for manual recovery."
            }
        }
        throw $launchError
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $stateReservation = $null
    try {
        $configuration = Get-ServiceLaunchConfiguration -BundleRoot $BundleRoot -BindHost $BindHost -Port $Port `
            -StateFile $StateFile -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost `
            -AudioTempRoot $AudioTempRoot -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio `
            -RubricPath $RubricPath -TracePath $TracePath
        $stateReservation = Acquire-StateReservation -StatePath $configuration.resolved_state_file
        Start-CompetitionEmotionService -Configuration $configuration -StateReservation $stateReservation `
            -ReplaceStaleState:$ReplaceStaleState `
            -AudioProxyUrl $AudioProxyUrl -AudioAllowedHost $AudioAllowedHost -AudioTempRoot $AudioTempRoot `
            -AudioBudgetSeconds $AudioBudgetSeconds -MaxConcurrentAudio $MaxConcurrentAudio `
            -RubricPath $RubricPath -TracePath $TracePath
    }
    finally {
        if ($null -ne $stateReservation) {
            $stateReservation.Dispose()
        }
    }
}
