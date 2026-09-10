# Official protocol service runbook

## Request chain and boundary

`client -> nonloopback listener -> FastAPI request validation -> lyrics/title model -> optional audio acquisition -> response`.

The service loads the immutable bundle selected by `BundleRoot\\current.json` at startup. It returns the two highest ranked labels from the lyrics/title model. Audio measurement may add evidence only; it does not change those ranked labels. The two models are the lyrics/title classifier and the bounded audio-measurement path. Their results have the formal baseline limitations described in [`official-self-evaluation.md`](official-self-evaluation.md); no production accuracy, latency, or availability claim follows from that evaluation.

Direct audio mode resolves and accepts only security-pinned public addresses. Proxy mode requires both a proxy origin and an exact allowlist of permitted DNS hosts. Do not use a proxy without the exact allowlist.

## Clean-host deployment

1. Install 64-bit Python 3.12 and confirm `py -3.12 --version` succeeds.
2. Install `ffmpeg` and confirm `ffmpeg -version` succeeds from the service account's environment.
3. From the repository checkout, install the service package and its locked dependencies: `py -3.12 -m pip install -e .\\competition_service`.
4. Copy `competition_service\\scripts\\service.env.example` to an operator-managed location. Put any proxy URL in a secret store or the process environment; never commit it.
5. Train or provision a validated immutable bundle. Confirm the chosen `BundleRoot` contains a regular `current.json` file and its active version contains `model.joblib` and `report.json`.
6. Choose a nonloopback IP literal that is routed and protected by the host or network firewall. `127.0.0.1`, `::1`, and `localhost` are rejected by both the launcher and direct Python CLI. `--host` is required by the Python CLI; it has no default. Passing `0.0.0.0` is allowed only when the operator explicitly supplies it.

Start a service on a private interface. Substitute the actual bundle path, IP, and allowlisted hosts:

```powershell
& .\\competition_service\\scripts\\start_service.ps1 `
  -BundleRoot 'D:\\competition\\official-bundle' `
  -BindHost '10.20.30.40' `
  -Port 8000 `
  -AudioProxyUrl $env:AUDIO_PROXY_URL `
  -AudioAllowedHost 'media.example.com','cdn.example.com' `
  -AudioTempRoot 'D:\\competition\\audio-tmp' `
  -AudioBudgetSeconds 20 `
  -MaxConcurrentAudio 4
```

The launcher starts a hidden child process, creates `logs\\service` under the bundle root, and atomically writes `service-state.json` only after the PID and its machine-parseable process creation time are present. The state file contains only PID, bind host, port, start time, and bundle root. It never stores the proxy URL or request URLs. A live owned state file blocks a second start. A stale or invalid state requires an operator review and an explicit `-ReplaceStaleState` on the next start; launch failures never remove a pre-existing state file.

Health-check locally from an authorized management host:

```powershell
Invoke-WebRequest -UseBasicParsing http://10.20.30.40:8000/healthz
# or: curl.exe http://10.20.30.40:8000/healthz
```

Expect HTTP 200 and a JSON payload with `ready: true`, the bundle model type, model version, and label count. Open the network firewall only for known caller networks; do not expose the listener broadly without the required perimeter controls.

## Configuration

| Setting | Meaning | Operator rule |
| --- | --- | --- |
| `BundleRoot` | Bundle pointer and version directory | Must exist and contain `current.json`. |
| `BindHost` | Listener address | Required nonloopback IP literal; no DNS name. |
| `Port` | TCP listener | Integer from 1 to 65535; default 8000. |
| `StateFile` | Restart ownership record | Default is `BundleRoot\\service-state.json`. |
| `AudioProxyUrl` | Optional trusted proxy origin | Must be paired with `AudioAllowedHost`; keep out of logs and source control. |
| `AudioAllowedHost` | Exact permitted audio origins | One or more DNS names; no IPs, paths, or wildcards. |
| `AudioTempRoot` | Disposable audio workspace | Use a writable local volume with a cleanup policy. |
| `AudioBudgetSeconds` | Per-request audio budget | Greater than 0 and no more than 25; default 20. |
| `MaxConcurrentAudio` | Non-queuing audio capacity | Positive integer; tune using saturation data. |

Restart with the same explicit nonsecret settings. The restart script reads the state file but refuses to stop a PID unless the process command line identifies `competition_emotion.service`, has exactly one matching bundle root, host, and port argument, and its Windows process creation time matches the recorded start time.

```powershell
& .\\competition_service\\scripts\\restart_service.ps1 `
  -BundleRoot 'D:\\competition\\official-bundle' `
  -BindHost '10.20.30.40' `
  -Port 8000
```

## Logs, monitoring, and retention

Each launch writes separate stdout and stderr files under `BundleRoot\\logs\\service`. Restrict access to service operators, because an upstream dependency can emit sensitive operational detail. Rotate by age and size; a practical starting point is 14 days or 1 GiB total, adjusted to your incident-retention policy. Remove old logs with the approved host maintenance job, never by deleting the active state file.

Collect and alert on QPS, end-to-end latency, HTTP error rate, CPU, memory, audio availability (`measured` versus `unavailable` evidence), and audio capacity saturation. Break latency down by request validation, model scoring, download, and ffmpeg decode when instrumentation is added. Alert on a sustained rise in unavailable audio, saturation, failures, or restart loops; health checks alone do not show request quality.

## FAQ

| Symptom | Likely cause | Recovery |
| --- | --- | --- |
| audio download fail or audio is unavailable | URL expired, target is not publicly routable, or the download budget elapsed | Verify a fresh HTTPS URL, DNS/public-address policy, and the configured audio budget. The ranked result still comes from lyrics/title. |
| proxy allowlist rejection | Proxy URL and allowlist were not supplied together, or the destination host is not an exact allowlisted DNS name | Supply both settings, remove paths/wildcards/IPs from the host list, and include the exact destination host. |
| model loading failure at startup | `current.json`, the active bundle, report provenance, or `model.joblib` is missing or invalid | Stop the failed launch, restore a complete immutable bundle, validate the active report, then start again. |
| OOM, high CPU, or growing memory | Concurrent requests/audio decodes exceed host capacity | Lower `MaxConcurrentAudio`, add CPU/memory capacity, and investigate ffmpeg and model process usage before retrying. |
| port collision | Another program owns the chosen listener | Identify the owner, stop only the verified intended service or choose an approved free port, then restart. |
| timed-out request or saturation | Audio work exceeds its deadline or all non-queuing audio permits are busy | Check upstream audio availability, lower traffic or raise capacity only after resource testing, then use the verified restart command if the service is unhealthy. |
| restart refused | State file is malformed or refers to a different process/bundle | Do not force-kill the recorded PID. Inspect the state, process command line, bundle path, and logs; repair the operator configuration before retrying. |
