# Official protocol service runbook

## Request chain and boundary

`client -> nonloopback listener -> FastAPI request validation -> metadata-aware local model -> optional candidate-limited semantic review -> optional audio acquisition -> response`.

The service loads the immutable bundle selected by `BundleRoot\\current.json` at startup. The local model ranks all 15 labels and returns one label plus a second candidate. When an optional semantic review URL is configured and the local margin is below the configured threshold, the reviewer may choose only from the adaptive local candidate pool (up to Top-10). The reviewer also receives the versioned soft Rubric context and must return a lyric quote that occurs in the supplied lyrics plus any Rubric rule IDs it used. An invalid, slow, or unavailable reviewer result falls back to the local ranking. The two models in the complete chain are the local classifier and the optional semantic reviewer; audio measurement is a bounded evidence path and does not change ranked labels. The formal held-out limitations are recorded in [`official-self-evaluation.md`](official-self-evaluation.md); no hidden-set accuracy claim follows from that evaluation.

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
  # Optional: -RubricPath 'D:\\competition\\emotion_rubric.json' -TracePath 'D:\\competition\\traces\\emotion.jsonl'
```

The launcher canonicalizes the supplied IP literal before passing it to Python and storing it in state (for example, `0` becomes `0.0.0.0`). It starts a hidden child process, creates `logs\\service` under the bundle root, and atomically writes `service-state.json` only after the PID and its machine-parseable process creation time are present. The state file contains only PID, canonical bind host, port, start time, and bundle root. It never stores the proxy URL or request URLs. An OS-level exclusive reservation for the state path is held from state inspection through publication, so concurrent launchers cannot replace one another's state. A live owned state file blocks a second start. A stale or invalid state requires an operator review and an explicit `-ReplaceStaleState` on the next start; the launcher revalidates the exact stale file, atomically renames it to a unique same-directory `.stale` backup before launch, removes that backup only after successful publication, and restores it on pre-publication failure when the state path remains absent and the backup is unchanged. If a newer state appears, or backup ownership cannot be verified, the launcher leaves the state path and backup untouched for manual recovery.

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
| `RubricPath` / `EMOTION_RUBRIC_PATH` | Versioned candidate definitions and evidence rules | Defaults to the repository Rubric; custom files must cover exactly the 15 official labels. |
| `TracePath` / `EMOTION_TRACE_PATH` | Append-only prediction trace for hard-case mining | Optional; keep outside the bundle and restrict access to operators. |
| `EMOTION_SEMANTIC_RERANKER_URL` | Optional internal candidate reviewer | Set only to an approved internal HTTP(S) endpoint (`.internal`, `.local`, `.corp`, or private IP); leave empty to disable. |
| `--semantic-reranker-timeout-seconds` | Reviewer timeout | 0.1–10 seconds; keep below the overall 25-second request budget. |
| `--semantic-reranker-min-gap` | Local score gap below which review runs | Default 0.10; calibrate on a separate validation set. |
| `--semantic-reranker-candidate-count` | Upper bound for local candidates supplied to the reviewer | Default 10; valid range 2–15. The service adapts this upper bound by score gap: below 0.03 uses the configured bound, 0.03–0.06 uses at most 7, and 0.06–0.10 uses at most 5. |

The reviewer endpoint is intentionally small and deterministic to integrate:
it receives the song fields, lyrics, the adaptive local candidate labels, and their
Rubric entries; it must return exactly one candidate with `confidence`, a short
`evidence` string, at least one verbatim `quotes` item found in the supplied
lyrics, and the corresponding `rule_ids`. Any schema, candidate, quote, or
rule mismatch is rejected and the local result is retained.

The Dev/calibration manifest is created offline from the fixed Train portion:

```powershell
py -3.12 -m competition_emotion.calibration `
  --workbook 'F:\\netease\\_music\\competition\\data\\emotion_songs_20260910.xlsx' `
  --output 'F:\\netease\\_music\\competition\\runs\\official-20260910\\calibration-20260911.json'
```

The accepted v9 bundle selects the model on the fixed holdout without the
54-row older snapshot augmentation, then retrains the published serving model
on all official songs. The accepted model also uses the operator-managed public
playlist snapshot whose SHA-256 is recorded in `report.json`. The reproducible
command is:

```powershell
.\competition_service\scripts\train_official.ps1 `
  -PlaylistPrior 'F:\netease\_music\competition\research_cache\netease_playlists\playlists_hot_500.json' `
  -FitAllForServing
```

An older augmentation workbook may still be supplied for a research candidate
with `-AugmentWorkbook`; it is never mixed into the accepted v9 run by default.

Use `read_traces` and `mine_hard_cases` to build a review queue, then evaluate a
proposed Rubric patch on Dev. The historical fixed Test is development-exposed
after repeated comparisons and can only be a regression check; it is not a
new blind test. See `docs/2026-09-16-evaluation-audit.md`.
The patch generator marks proposals as `proposed` and requires human review;
there is no online rule mutation.

After the Dev gate and historical regression check pass and a reviewer approves the change,
`publish_rubric_candidate` atomically replaces the operator-selected Rubric
file. A failed gate or missing approval cannot publish.

```powershell
py -3.12 -m competition_emotion.iteration `
  --trace-path 'F:\\netease\\_music\\competition\\runs\\official-20260910\\traces\\emotion.jsonl' `
  --output 'F:\\netease\\_music\\competition\\runs\\official-20260910\\rubric-patch-proposal.json'
```

Restart with the same explicit nonsecret settings. The restart script canonicalizes its IP literal before comparison, then refuses to stop a PID unless the process command line identifies `competition_emotion.service`, has exactly one matching bundle root, canonical host, and port argument, and its Windows process creation time matches the recorded start time. It holds the same OS-level state reservation used by start from the state check through process stop, state removal, and publication of the replacement state, so a concurrent stale-state start waits for the completed restart.

```powershell
& .\\competition_service\\scripts\\restart_service.ps1 `
  -BundleRoot 'D:\\competition\\official-bundle' `
  -BindHost '10.20.30.40' `
  -Port 8000
```

## Logs, monitoring, and retention

Each launch writes separate stdout and stderr files under `BundleRoot\\logs\\service`. Restrict access to service operators, because an upstream dependency can emit sensitive operational detail. Rotate by age and size; a practical starting point is 14 days or 1 GiB total, adjusted to your incident-retention policy. Remove old logs with the approved host maintenance job, never by deleting the active state file.

Collect and alert on QPS, end-to-end latency, HTTP error rate, CPU, memory, audio download/decode failures (HTTP 502), and audio capacity saturation. Break latency down by request validation, model scoring, download, and ffmpeg decode when instrumentation is added. Alert on a sustained rise in audio failures, saturation, or restart loops; health checks alone do not show request quality.

## FAQ

| Symptom | Likely cause | Recovery |
| --- | --- | --- |
| audio download fail or audio is unavailable | URL expired, target is not publicly routable, or the download budget elapsed | The API returns HTTP 502 without a label. Verify a fresh HTTPS URL, DNS/public-address policy, and the configured audio budget. |
| proxy allowlist rejection | Proxy URL and allowlist were not supplied together, or the destination host is not an exact allowlisted DNS name | Supply both settings, remove paths/wildcards/IPs from the host list, and include the exact destination host. |
| model loading failure at startup | `current.json`, the active bundle, report provenance, or `model.joblib` is missing or invalid | Stop the failed launch, restore a complete immutable bundle, validate the active report, then start again. |
| OOM, high CPU, or growing memory | Concurrent requests/audio decodes exceed host capacity | Lower `MaxConcurrentAudio`, add CPU/memory capacity, and investigate ffmpeg and model process usage before retrying. |
| port collision | Another program owns the chosen listener | Identify the owner, stop only the verified intended service or choose an approved free port, then restart. |
| timed-out request or saturation | Audio work exceeds its deadline or all non-queuing audio permits are busy | Check upstream audio availability, lower traffic or raise capacity only after resource testing, then use the verified restart command if the service is unhealthy. |
| semantic review unavailable | Reviewer endpoint timed out, returned a non-candidate, or was not configured | Check the internal endpoint and its response schema. The service safely uses the local Top-1 and records no unverified review evidence. |
| restart refused | State file is malformed or refers to a different process/bundle | Do not force-kill the recorded PID. Inspect the state, process command line, bundle path, and logs; repair the operator configuration before retrying. |
