"""Contract tests for the operator-facing PowerShell and runbook artifacts."""

from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SERVICE_ROOT / "scripts"
DOCS = SERVICE_ROOT.parents[0] / "docs"


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
        self.assertRegex(source, r"\[ordered\]@\{[\s\S]*pid[\s\S]*bind_host[\s\S]*port[\s\S]*start_time_utc[\s\S]*bundle_root")
        self.assertNotIn("audio_proxy_url", source.lower())

    def test_restart_script_refuses_untrusted_state_before_stopping_a_process(self) -> None:
        source = read(SCRIPTS / "restart_service.ps1")

        self.assertIn("ConvertFrom-Json", source)
        self.assertIn("state schema", source)
        self.assertIn("Get-CimInstance Win32_Process", source)
        self.assertRegex(source, r"competition_emotion\\+\.service")
        self.assertIn("Resolve-Path -LiteralPath $BundleRoot", source)
        self.assertIn("Stop-Process -Id $statePid -Force", source)
        self.assertLess(source.index("competition_emotion\\.service"), source.index("Stop-Process -Id $statePid -Force"))
        self.assertIn("& $startScript", source)

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
