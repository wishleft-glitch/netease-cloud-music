"""Contract tests for the operator-facing PowerShell and runbook artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from competition_emotion import service


SERVICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SERVICE_ROOT / "scripts"
DOCS = SERVICE_ROOT.parents[0] / "docs"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def read(relative: Path) -> str:
    return relative.read_text(encoding="utf-8")


class ServiceOperationsArtifactTests(unittest.TestCase):
    def test_start_script_requires_a_non_loopback_ip_and_a_valid_port(self) -> None:
        source = read(SCRIPTS / "start_service.ps1")

        self.assertRegex(source, r"\[Parameter\(Mandatory\)\][\s\S]{0,120}\[string\]\$BundleRoot")
        self.assertRegex(source, r"\[Parameter\(Mandatory\)\][\s\S]{0,120}\[string\]\$BindHost")
        self.assertIn("[System.Net.IPAddress]::TryParse", source)
        self.assertIn("IPAddress]::IsLoopback", source)
        self.assertRegex(source, r"-not \(\$Port -is \[int\]\) -or \$Port -lt 1 -or \$Port -gt 65535")
        self.assertNotIn('default="127.0.0.1"', source)
        self.assertNotIn('default="localhost"', source)

    def test_direct_service_cli_requires_a_nonloopback_ip_literal(self) -> None:
        with patch.object(service, "create_app"), patch.object(service.uvicorn, "run"):
            with self.assertRaises(SystemExit) as missing_host:
                service.main(["--bundle-root", "bundle"])
            with self.assertRaises(SystemExit) as loopback_host:
                service.main(["--bundle-root", "bundle", "--host", "localhost"])
            with self.assertRaises(SystemExit) as dns_host:
                service.main(["--bundle-root", "bundle", "--host", "service.example.test"])

        self.assertEqual(missing_host.exception.code, 2)
        self.assertEqual(loopback_host.exception.code, 2)
        self.assertEqual(dns_host.exception.code, 2)

        with patch.object(service, "create_app") as create_app, patch.object(service.uvicorn, "run") as run:
            service.main(["--bundle-root", "bundle", "--host", "0.0.0.0"])

        self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(create_app.call_args.args[0], Path("bundle"))

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise the launcher helper")
    def test_powershell_option_parser_reads_the_launchers_quoted_host_and_port(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        command = f'''$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({str(script_path)!r}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ exit 2 }}
$function = $ast.Find({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq "Get-CommandLineOptionValues" }}, $true)
. ([ScriptBlock]::Create($function.Extent.Text))
$line = '"-3.12" "-m" "competition_emotion.service" "--bundle-root" "C:\\Bundle Root" "--host" "10.20.30.40" "--port" "8000"'
$hosts = @(Get-CommandLineOptionValues -CommandLine $line -Option "--host")
$ports = @(Get-CommandLineOptionValues -CommandLine $line -Option "--port")
if ($hosts.Count -ne 1 -or $hosts[0] -cne "10.20.30.40" -or $ports.Count -ne 1 -or $ports[0] -cne "8000") {{ exit 1 }}'''
        result = subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise the launcher reservation")
    def test_powershell_launcher_reservation_is_exclusive_and_canonicalizes_zero(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        restart_script_path = SCRIPTS / "restart_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_file = Path(temporary_directory) / "service-state.json"
            command = f'''$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile({str(script_path)!r}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ exit 2 }}
foreach ($name in @("Acquire-StateReservation", "Get-CanonicalBindHost")) {{
    $function = $ast.Find({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }}, $true)
    if ($null -eq $function) {{ exit 3 }}
    . ([ScriptBlock]::Create($function.Extent.Text))
}}
if ((Get-CanonicalBindHost -HostValue "0") -cne "0.0.0.0") {{ exit 4 }}
$restartTokens = $null
$restartErrors = $null
$restartAst = [System.Management.Automation.Language.Parser]::ParseFile({str(restart_script_path)!r}, [ref]$restartTokens, [ref]$restartErrors)
if ($restartErrors.Count) {{ exit 7 }}
$restartFunction = $restartAst.Find({{ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq "Get-CanonicalBindHost" }}, $true)
if ($null -eq $restartFunction) {{ exit 8 }}
. ([ScriptBlock]::Create($restartFunction.Extent.Text))
if ((Get-CanonicalBindHost -HostValue "0") -cne "0.0.0.0") {{ exit 9 }}
$reservation = Acquire-StateReservation -StatePath {str(state_file)!r}
try {{
    if (-not $reservation.Name.EndsWith(".launch.lock", [System.StringComparison]::Ordinal)) {{ Write-Output $reservation.Name; exit 6 }}
    try {{
        $second = [System.IO.File]::Open($reservation.Name, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        $second.Dispose()
        exit 5
    }}
    catch [System.IO.IOException] {{ }}
}}
finally {{
    $reservation.Dispose()
}}'''
            result = subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise the launcher state parser")
    def test_powershell_restart_accepts_launcher_iso_state_before_safe_identity_handoff(self) -> None:
        start_script_path = SCRIPTS / "start_service.ps1"
        restart_script_path = SCRIPTS / "restart_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundle_root.mkdir()
            (bundle_root / "current.json").write_text("{}", encoding="utf-8")
            state_file = temporary_root / "service-state.json"
            command = f'''$ErrorActionPreference = "Stop"
trap {{ [Console]::Error.WriteLine("STATE-PARSER-TEST: " + $_.Exception.Message); exit 97 }}
$bundleRoot = {str(bundle_root)!r}
$stateFile = {str(state_file)!r}
$script:expectedCreationTime = "2030-01-02T03:04:05.0000000Z"
$stateJson = '{{"pid":424242,"bind_host":"10.20.30.40","port":8000,"start_time_utc":"' + $script:expectedCreationTime + '","bundle_root":' + ($bundleRoot | ConvertTo-Json -Compress) + '}}'
[System.IO.File]::WriteAllText($stateFile, $stateJson, [System.Text.UTF8Encoding]::new($false))
$script:stoppedPid = $null
$script:launchCalled = $false
function Get-CimInstance {{
    [CmdletBinding()]
    param([string]$ClassName, [string]$Filter)
    if ($Filter -match "424242") {{
        return [pscustomobject]@{{
            CommandLine = 'py -3.12 -m competition_emotion.service --bundle-root "' + $bundleRoot + '" --host 10.20.30.40 --port 8000'
            CreationDate = [DateTime]::ParseExact($script:expectedCreationTime, "o", [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
        }}
    }}
    if ($Filter -match "434343") {{
        return [pscustomobject]@{{ CreationDate = [DateTime]::ParseExact($script:expectedCreationTime, "o", [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind) }}
    }}
    return $null
}}
function Stop-Process {{
    [CmdletBinding()]
    param([int]$Id, $InputObject, [switch]$Force)
    if ($PSBoundParameters.ContainsKey("Id")) {{ $script:stoppedPid = $Id }}
}}
function Wait-Process {{ [CmdletBinding()] param([int]$Id, [int]$Timeout) }}
function Get-Process {{
    [CmdletBinding()]
    param([int]$Id)
    if ($Id -eq 434343) {{ return [pscustomobject]@{{ Id = 434343; HasExited = $false }} }}
    return $null
}}
function Start-Process {{
    [CmdletBinding()]
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    $script:launchCalled = $true
    return [pscustomobject]@{{ Id = 434343 }}
}}
function Start-Sleep {{ param([int]$Milliseconds) }}

. {str(restart_script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile 6>$null
if ($script:stoppedPid -ne 424242 -or -not $script:launchCalled) {{
    throw "Restart did not reach the safe stop and replacement launch path."
}}

[System.IO.File]::WriteAllText($stateFile, $stateJson, [System.Text.UTF8Encoding]::new($false))
. {str(start_script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile
$snapshot = Get-StateSnapshot -Path $stateFile
if ((Get-ExistingStateStatus -Snapshot $snapshot) -cne "LiveOwned") {{
    throw "Start classified a valid owned state as stale."
}}

function Get-Command {{
    [CmdletBinding()]
    param([string]$Name)
    if ($Name -eq "ConvertFrom-Json") {{
        return [pscustomobject]@{{ Parameters = @{{}} }}
    }}
    return Microsoft.PowerShell.Core\\Get-Command @PSBoundParameters
}}
if ((ConvertFrom-ServiceStateJson -Content $stateJson).start_time_utc -cne $script:expectedCreationTime) {{
    throw "The legacy PowerShell parser fallback did not preserve the canonical timestamp."
}}

$nonUtcJson = $stateJson.Replace($script:expectedCreationTime, "2030-01-02T11:04:05.0000000+08:00")
try {{
    ConvertFrom-ServiceStateJson -Content $nonUtcJson | Out-Null
    throw "Non-UTC timestamp was accepted."
}}
catch {{
    if ($_.Exception.Message -notmatch "Invalid state schema") {{ throw }}
}}
$malformedJson = $stateJson.Replace($script:expectedCreationTime, "2030-01-02T03:04:05Z")
try {{
    ConvertFrom-ServiceStateJson -Content $malformedJson | Out-Null
    throw "Malformed timestamp was accepted."
}}
catch {{
    if ($_.Exception.Message -notmatch "Invalid state schema") {{ throw }}
}}
exit 0'''
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}\n" + result.stderr + result.stdout)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise stale-state launch recovery")
    def test_powershell_launch_failure_restores_the_exact_stale_state(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundle_root.mkdir()
            (bundle_root / "current.json").write_text("{}", encoding="utf-8")
            state_file = temporary_root / "service-state.json"
            old_state = '{"legacy":"old"}'
            command = f'''$ErrorActionPreference = "Stop"
$bundleRoot = {str(bundle_root)!r}
$stateFile = {str(state_file)!r}
$stateDirectory = {str(temporary_root)!r}
$oldState = {old_state!r}
[System.IO.File]::WriteAllText($stateFile, $oldState, [System.Text.UTF8Encoding]::new($false))
function global:Start-Process {{
    [CmdletBinding()]
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    throw "synthetic launch failure"
}}
$failed = $false
try {{
    & {str(script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile -ReplaceStaleState
}}
catch {{
    $failed = $true
}}
if (-not $failed) {{ throw "Expected the synthetic launch failure." }}
if (-not (Test-Path -LiteralPath $stateFile -PathType Leaf)) {{ throw "The stale state path was not restored." }}
if ([System.IO.File]::ReadAllText($stateFile) -cne $oldState) {{ throw "The restored state content changed." }}
$backups = @(Get-ChildItem -LiteralPath $stateDirectory -Force | Where-Object {{ $_.Name -like ".service-state.json.*.stale" }})
if ($backups.Count -ne 0) {{ throw "The stale backup should be consumed after restoration." }}
exit 0'''
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}\n" + result.stderr + result.stdout)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise stale-backup ownership recovery")
    def test_powershell_launch_failure_does_not_restore_a_modified_stale_backup(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundle_root.mkdir()
            (bundle_root / "current.json").write_text("{}", encoding="utf-8")
            state_file = temporary_root / "service-state.json"
            old_state = '{"legacy":"old"}'
            modified_state = '{"tampered":"backup"}'
            command = f'''$ErrorActionPreference = "Stop"
$bundleRoot = {str(bundle_root)!r}
$stateFile = {str(state_file)!r}
$stateDirectory = {str(temporary_root)!r}
$modifiedState = {modified_state!r}
[System.IO.File]::WriteAllText($stateFile, {old_state!r}, [System.Text.UTF8Encoding]::new($false))
function global:Start-Process {{
    [CmdletBinding()]
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    $backup = Get-ChildItem -LiteralPath $stateDirectory -Force | Where-Object {{ $_.Name -like ".service-state.json.*.stale" }} | Select-Object -First 1
    if ($null -eq $backup) {{ throw "Synthetic launch could not find the stale backup." }}
    [System.IO.File]::WriteAllText($backup.FullName, $modifiedState, [System.Text.UTF8Encoding]::new($false))
    throw "synthetic launch failure"
}}
$failed = $false
try {{
    & {str(script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile -ReplaceStaleState
}}
catch {{
    $failed = $true
}}
if (-not $failed) {{ throw "Expected the synthetic launch failure." }}
if (Test-Path -LiteralPath $stateFile -PathType Leaf) {{ throw "A modified stale backup must not be restored as state." }}
$backups = @(Get-ChildItem -LiteralPath $stateDirectory -Force | Where-Object {{ $_.Name -like ".service-state.json.*.stale" }})
if ($backups.Count -ne 1) {{ throw "The modified stale backup was not retained for manual recovery." }}
if ([System.IO.File]::ReadAllText($backups[0].FullName) -cne $modifiedState) {{ throw "The modified stale backup was changed after ownership was lost." }}
exit 0'''
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}\n" + result.stderr + result.stdout)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise stale-state publication recovery")
    def test_powershell_publication_failure_restores_the_exact_stale_state(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundle_root.mkdir()
            (bundle_root / "current.json").write_text("{}", encoding="utf-8")
            state_file = temporary_root / "service-state.json"
            old_state = '{"legacy":"old"}'
            command = f'''$ErrorActionPreference = "Stop"
$bundleRoot = {str(bundle_root)!r}
$stateFile = {str(state_file)!r}
$stateDirectory = {str(temporary_root)!r}
$oldState = {old_state!r}
$global:MoveCallCount = 0
$global:RecoveryBundleRoot = $bundleRoot
[System.IO.File]::WriteAllText($stateFile, $oldState, [System.Text.UTF8Encoding]::new($false))
function global:Start-Process {{
    [CmdletBinding()]
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    return [pscustomobject]@{{ Id = 434343 }}
}}
function global:Get-Process {{
    [CmdletBinding()]
    param([int]$Id)
    if ($Id -eq 434343) {{ return [pscustomobject]@{{ Id = 434343; HasExited = $false }} }}
    return $null
}}
function global:Get-CimInstance {{
    [CmdletBinding()]
    param([string]$ClassName, [string]$Filter)
    return [pscustomobject]@{{
        CommandLine = 'py -3.12 -m competition_emotion.service --bundle-root "' + $global:RecoveryBundleRoot + '" --host 10.20.30.40 --port 8000'
        CreationDate = [DateTime]::ParseExact("2030-01-02T03:04:05.0000000Z", "o", [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
    }}
}}
function global:Stop-Process {{
    [CmdletBinding()]
    param([int]$Id, $InputObject, [switch]$Force)
}}
function global:Move-Item {{
    [CmdletBinding()]
    param([string]$LiteralPath, [string]$Destination, [switch]$Force)
    $global:MoveCallCount += 1
    if ($global:MoveCallCount -eq 2) {{
        throw "synthetic publication failure"
    }}
    Microsoft.PowerShell.Management\\Move-Item @PSBoundParameters
}}
$failed = $false
try {{
    & {str(script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile -ReplaceStaleState
}}
catch {{
    $failed = $true
}}
if (-not $failed) {{ throw "Expected the synthetic publication failure." }}
if (-not (Test-Path -LiteralPath $stateFile -PathType Leaf)) {{ throw "The stale state path was not restored." }}
if ([System.IO.File]::ReadAllText($stateFile) -cne $oldState) {{ throw "The restored state content changed." }}
$backups = @(Get-ChildItem -LiteralPath $stateDirectory -Force | Where-Object {{ $_.Name -like ".service-state.json.*.stale" }})
if ($backups.Count -ne 0) {{ throw "The stale backup should be consumed after restoration." }}
exit 0'''
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}\n" + result.stderr + result.stdout)

    @unittest.skipUnless(POWERSHELL, "PowerShell is required to exercise stale-state concurrency recovery")
    def test_powershell_new_state_during_launch_failure_is_preserved_with_backup(self) -> None:
        script_path = SCRIPTS / "start_service.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundle_root.mkdir()
            (bundle_root / "current.json").write_text("{}", encoding="utf-8")
            state_file = temporary_root / "service-state.json"
            old_state = '{"legacy":"old"}'
            new_state = '{"new":"state"}'
            command = f'''$ErrorActionPreference = "Stop"
$bundleRoot = {str(bundle_root)!r}
$stateFile = {str(state_file)!r}
$stateDirectory = {str(temporary_root)!r}
$oldState = {old_state!r}
$newState = {new_state!r}
$global:RecoveryStateFile = $stateFile
$global:RecoveryNewState = $newState
$global:RecoveryBundleRoot = $bundleRoot
[System.IO.File]::WriteAllText($stateFile, $oldState, [System.Text.UTF8Encoding]::new($false))
function global:Start-Process {{
    [CmdletBinding()]
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    [System.IO.File]::WriteAllText($global:RecoveryStateFile, $global:RecoveryNewState, [System.Text.UTF8Encoding]::new($false))
    return [pscustomobject]@{{ Id = 434343 }}
}}
function global:Get-Process {{
    [CmdletBinding()]
    param([int]$Id)
    if ($Id -eq 434343) {{ return [pscustomobject]@{{ Id = 434343; HasExited = $false }} }}
    return $null
}}
function global:Get-CimInstance {{
    [CmdletBinding()]
    param([string]$ClassName, [string]$Filter)
    return [pscustomobject]@{{
        CommandLine = 'py -3.12 -m competition_emotion.service --bundle-root "' + $global:RecoveryBundleRoot + '" --host 10.20.30.40 --port 8000'
        CreationDate = [DateTime]::ParseExact("2030-01-02T03:04:05.0000000Z", "o", [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
    }}
}}
function global:Stop-Process {{
    [CmdletBinding()]
    param([int]$Id, $InputObject, [switch]$Force)
}}
$failed = $false
try {{
    & {str(script_path)!r} -BundleRoot $bundleRoot -BindHost "10.20.30.40" -Port 8000 -StateFile $stateFile -ReplaceStaleState
}}
catch {{
    $failed = $true
}}
if (-not $failed) {{ throw "Expected the newer-state guard failure." }}
if ([System.IO.File]::ReadAllText($stateFile) -cne $newState) {{ throw "The newer state was overwritten." }}
$backups = @(Get-ChildItem -LiteralPath $stateDirectory -Force | Where-Object {{ $_.Name -like ".service-state.json.*.stale" }})
if ($backups.Count -ne 1) {{ throw "The stale backup was not retained for explicit recovery." }}
if ([System.IO.File]::ReadAllText($backups[0].FullName) -cne $oldState) {{ throw "The retained stale backup content changed." }}
exit 0'''
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}\n" + result.stderr + result.stdout)

    def test_start_script_validates_bundle_and_keeps_proxy_out_of_logs_and_state(self) -> None:
        source = read(SCRIPTS / "start_service.ps1")

        self.assertIn('Join-Path $resolvedBundleRoot "current.json"', source)
        self.assertIn("-PathType Leaf", source)
        self.assertIn("AudioProxyUrl requires at least one AudioAllowedHost", source)
        self.assertIn("AudioAllowedHost requires AudioProxyUrl", source)
        self.assertNotRegex(source, r"Write-(?:Host|Output|Information|Verbose)[^\r\n]*AudioProxyUrl")
        self.assertIn("-WindowStyle Hidden", source)
        self.assertIn("-RedirectStandardOutput", source)
        self.assertIn("-RedirectStandardError", source)
        self.assertIn("function ConvertTo-WindowsCommandLineArgument", source)
        self.assertIn("-ArgumentList $argumentLine", source)
        self.assertIn("Get-Process -Id $startedProcess.Id", source)
        self.assertIn("Move-Item -LiteralPath $temporaryStatePath -Destination $resolvedStateFile -Force", source)
        self.assertIn("[switch]$ReplaceStaleState", source)
        self.assertIn("Get-ExistingStateStatus", source)
        self.assertIn("Acquire-StateReservation", source)
        self.assertIn("Get-StateSnapshot", source)
        self.assertIn("Test-StateSnapshotUnchanged", source)
        self.assertIn("FileShare]::None", source)
        self.assertIn("$stateReservation.Dispose()", source)
        helper_source = source[source.index("function Start-CompetitionEmotionService"):]
        self.assertIn("[Parameter(Mandatory)][System.IO.FileStream]$StateReservation", helper_source)
        self.assertIn("matching reservation from the start or restart script", helper_source)
        self.assertIn("state_directory = $stateDirectory", source)
        self.assertIn("$stateDirectory = $Configuration.state_directory", helper_source)
        self.assertIn("$allowedCallerPaths", helper_source)
        self.assertIn("$MyInvocation.ScriptName", helper_source)
        self.assertIn("$existingStateSnapshot = Get-StateSnapshot", helper_source)
        main_source = source[source.rindex("$stateReservation = Acquire-StateReservation"):]
        self.assertLess(main_source.index("$stateReservation = Acquire-StateReservation"), main_source.index("Start-CompetitionEmotionService"))
        self.assertLess(source.index("Test-StateSnapshotUnchanged"), source.index("Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop"))
        self.assertIn('"--host", $canonicalBindHost', source)
        self.assertIn("bind_host = $canonicalBindHost", source)
        self.assertIn("A live owned service state already exists", source)
        self.assertIn("ReplaceStaleState", source)
        self.assertIn("Get-CimInstance Win32_Process", source)
        self.assertRegex(source, r"\[ordered\]@\{[\s\S]*pid[\s\S]*bind_host[\s\S]*port[\s\S]*start_time_utc[\s\S]*bundle_root")
        self.assertIn("CreationDate", source)
        self.assertIn("$canonicalBindHost", source)
        self.assertIn("Get-CanonicalBindHost", source)
        catch_block = source.rsplit("catch {", 1)[1]
        self.assertIn("if ($statePublishedByThisInvocation)", catch_block)
        self.assertNotIn("audio_proxy_url", source.lower())

    def test_restart_script_refuses_untrusted_state_before_stopping_a_process(self) -> None:
        source = read(SCRIPTS / "restart_service.ps1")

        self.assertIn("ConvertFrom-ServiceStateJson -Content", source)
        self.assertIn("state schema", source)
        self.assertIn("Get-CimInstance Win32_Process", source)
        self.assertRegex(source, r"competition_emotion\\+\.service")
        self.assertIn("Get-ServiceLaunchConfiguration -BundleRoot $BundleRoot", source)
        self.assertIn("Stop-Process -Id $statePid -Force", source)
        self.assertLess(source.index("competition_emotion\\.service"), source.index("Stop-Process -Id $statePid -Force"))
        self.assertIn("CreationDate", source)
        self.assertIn('Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--host"', source)
        self.assertIn('Get-CommandLineOptionValues -CommandLine $process.CommandLine -Option "--port"', source)
        self.assertIn("process creation time", source)
        self.assertIn("-BindHost $BindHost", source)
        self.assertIn("$canonicalBindHost = $configuration.canonical_bind_host", source)
        self.assertIn(". $startScript -BundleRoot", source)
        self.assertIn("Start-CompetitionEmotionService", source)
        self.assertIn("State PID did not stop. Refusing to remove state.", source)
        self.assertNotIn("& $startScript", source)

    def test_restart_reserves_the_start_lock_before_inspecting_state_and_reuses_it_for_publication(self) -> None:
        source = read(SCRIPTS / "restart_service.ps1")

        reservation = "$stateReservation = Acquire-StateReservation -StatePath $resolvedStateFile"
        self.assertIn(reservation, source)
        self.assertLess(source.index(reservation), source.index("Test-Path -LiteralPath $resolvedStateFile"))
        self.assertLess(source.index(reservation), source.index("Get-Content -LiteralPath $resolvedStateFile -Raw"))
        self.assertLess(source.index(reservation), source.index("Stop-Process -Id $statePid -Force"))
        self.assertLess(source.index(reservation), source.index("Remove-Item -LiteralPath $resolvedStateFile -Force -ErrorAction Stop"))
        self.assertIn("$stateReservation.Dispose()", source)
        self.assertIn("finally", source)
        self.assertIn("-StateReservation $stateReservation", source)
        self.assertNotIn("& $startScript", source)

    def test_environment_example_keeps_proxy_secret_empty_and_operator_supplied(self) -> None:
        source = read(SCRIPTS / "service.env.example")

        self.assertIn("AUDIO_PROXY_URL=", source)
        self.assertIn("AUDIO_ALLOWED_HOSTS=", source)
        self.assertIn("never committed", source.lower())
        self.assertNotRegex(source, r"AUDIO_PROXY_URL=https?://.+")

    def test_launcher_capacity_flag_is_supported_by_the_service_cli(self) -> None:
        launcher = read(SCRIPTS / "start_service.ps1")
        service = read(SERVICE_ROOT / "src" / "competition_emotion" / "service.py")

        self.assertIn('"--max-concurrent-audio", $MaxConcurrentAudio', launcher)
        self.assertIn('parser.add_argument("--max-concurrent-audio"', service)
        self.assertIn("max_concurrent_audio=arguments.max_concurrent_audio", service)

    def test_runbook_covers_startup_monitoring_and_required_recoveries(self) -> None:
        source = read(DOCS / "official-protocol-runbook.md").lower()

        for phrase in (
            "request chain",
            "ffmpeg",
            "healthz",
            "audio download fail",
            "proxy allowlist",
            "model loading",
            "oom",
            "port collision",
            "timed-out",
            "saturation",
            "qps",
            "latency",
            "error rate",
            "cpu",
            "memory",
            "direct audio mode",
            "security-pinned public addresses",
            "exact allowlist",
            "two models",
        ):
            self.assertIn(phrase, source)

    def test_cost_document_has_formulas_and_does_not_invent_percentiles(self) -> None:
        source = read(DOCS / "official-cost-estimate.md")

        self.assertIn("CPU core seconds × internal rate", source)
        self.assertIn("not enabled / 0", source)
        self.assertRegex(source, r"P50[\s\S]{0,80}unavailable")
        self.assertRegex(source, r"P99[\s\S]{0,80}unavailable")
        self.assertIn("<=0.1", source)
        self.assertNotRegex(source, r"P(?:50|99)\s*[:=]\s*\d+(?:\.\d+)?\s*(?:ms|s)")


if __name__ == "__main__":
    unittest.main()
