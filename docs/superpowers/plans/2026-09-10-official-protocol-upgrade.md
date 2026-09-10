# Official Emotion Protocol Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a synchronous internal HTTP service that conforms to the updated official evaluation protocol, retrains from the 2026-09-10 export, and produces reproducible evaluation and deployment material.

**Architecture:** Keep the trusted local model bundle as the prediction core. Add a protocol adapter that validates official request fields, normalizes all provided lyric variants, downloads and decodes audio under one bounded deadline, then returns a deterministic Top-2 response with evidence based only on the request’s actual usable inputs. Model bundle publication remains versioned and atomic; HTTP service runtime never trusts model locations supplied by a request.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, Uvicorn, scikit-learn, httpx/httpcore, ffmpeg, pandas/openpyxl, unittest.

---

## File structure

- `competition_service/src/competition_emotion/service.py`: official HTTP schema, envelope, request body cap, inference runtime and health endpoint.
- `competition_service/src/competition_emotion/evidence.py`: bounded, factual evidence rendering from normalized lyric and audio-stage facts.
- `competition_service/src/competition_emotion/request_audio.py`: request-scoped secure audio acquisition, decoding and cleanup under an absolute deadline.
- `competition_service/src/competition_emotion/data.py`: current formal export loader and canonical lyric merge.
- `competition_service/src/competition_emotion/train.py`: model/report version metadata and repeatable retraining entry point.
- `competition_service/tests/test_service.py`, `test_evidence.py`, `test_request_audio.py`, `test_data.py`, `test_train.py`: protocol and regression coverage.
- `competition_service/scripts/start_service.ps1`, `restart_service.ps1`, `service.env.example`: repeatable internal deployment operations.
- `docs/official-protocol-runbook.md`, `docs/official-cost-estimate.md`, `docs/official-self-evaluation.md`: required delivery material.

## Task 1: Formal export provenance and lyric normalization

**Files:**
- Modify: `competition_service/src/competition_emotion/data.py`
- Modify: `competition_service/tests/test_data.py`
- Create: `competition_service/src/competition_emotion/lyrics.py`
- Create: `competition_service/tests/test_lyrics.py`

- [ ] **Step 1: Write failing loader and lyric-composition tests.**

```python
def test_compose_lyrics_prefers_all_nonempty_official_fields() -> None:
    text = compose_lyrics("plain", "[00:01]滚词", "[00:01]translation")
    self.assertEqual(text, "plain\n滚词\ntranslation")


def test_formal_export_provenance_contains_hash_and_song_count() -> None:
    provenance = workbook_provenance(self.workbook)
    self.assertEqual(provenance["sha256"], EXPECTED_SHA256)
    self.assertEqual(provenance["rows"], 5894)
```

- [ ] **Step 2: Run the new tests and verify they fail because `compose_lyrics` and `workbook_provenance` do not exist.**

Run: `py -3.12 -m unittest competition_service.tests.test_lyrics competition_service.tests.test_data -v`  
Expected: failure naming the missing functions.

- [ ] **Step 3: Implement canonical lyric composition and source provenance.**

```python
def compose_lyrics(text_lyric: object, lrc_lyric: object, lrc_translation: object) -> str:
    parts = [clean_lyric(value) for value in (text_lyric, lrc_lyric, lrc_translation)]
    return "\n".join(dict.fromkeys(part for part in parts if part))


def workbook_provenance(path: Path) -> dict[str, object]:
    return {"file_name": path.name, "sha256": sha256_file(path), "rows": int(len(pd.read_excel(path)))}
```

Update `load_official_songs` to call `compose_lyrics`, preserving deterministic song grouping and all 15 configured labels.

- [ ] **Step 4: Run the targeted tests and then the full suite.**

Run: `py -3.12 -m unittest discover -s competition_service/tests -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit.**

```powershell
git add competition_service/src/competition_emotion/data.py competition_service/src/competition_emotion/lyrics.py competition_service/tests/test_data.py competition_service/tests/test_lyrics.py
git commit -m "feat: normalize official lyric inputs"
```

## Task 2: Official synchronous request and response envelope

**Files:**
- Modify: `competition_service/src/competition_emotion/service.py`
- Modify: `competition_service/tests/test_service.py`
- Create: `competition_service/src/competition_emotion/evidence.py`
- Create: `competition_service/tests/test_evidence.py`

- [ ] **Step 1: Write failing official-contract tests.**

```python
def test_official_request_returns_top2_envelope_and_factual_lyrics_evidence(self) -> None:
    response = self.client.post("/api/v1/emotion/recognize", json={
        "audio_url": "https://audio.example/song.mp3", "song_name": "新歌", "song_id": "42",
        "artists": "歌手", "text_lyric": "我想念你",
    })
    self.assertEqual(response.status_code, 200)
    data = response.json()["data"]
    self.assertEqual(response.json()["code"], 200)
    self.assertNotEqual(data["top_emotion"], data["second_emotion"])
    self.assertRegex(str(data["top_confidence"]), r"^0(?:\.\d{1,4})?$|^1(?:\.0{1,4})?$")
    self.assertLessEqual(len(data["evidence"]), 500)


def test_missing_audio_url_returns_protocol_error_envelope(self) -> None:
    response = self.client.post("/api/v1/emotion/recognize", json={"song_name": "x", "song_id": "1"})
    self.assertEqual(response.status_code, 400)
    self.assertEqual(response.json()["code"], 400)
```

- [ ] **Step 2: Run the targeted service tests and verify failure against the obsolete `title` contract.**

Run: `py -3.12 -m unittest competition_service.tests.test_service -v`  
Expected: failure because the response lacks `code/data/error` and official fields.

- [ ] **Step 3: Implement strict models and an explicit error handler.**

```python
class OfficialRecognizeRequest(BaseModel):
    audio_url: StrictStr
    song_name: StrictStr
    song_id: StrictStr
    album_name: StrictStr | None = None
    artists: StrictStr | None = None
    text_lyric: StrictStr | None = None
    lrc_lyric: StrictStr | None = None
    lrc_translation: StrictStr | None = None


def success_response(top: RankedEmotion, second: RankedEmotion, evidence: str, cost_ms: int) -> dict[str, object]:
    return {"code": 200, "data": {...}, "error": ""}
```

Rank every configured label before selecting the first two, round only serialized confidences to four decimals, and reject any invalid score before ranking. Add a FastAPI validation-error handler that returns `{"code":400,"message":"invalid request"}` without local paths or model details. Keep all-score and request-size defenses.

- [ ] **Step 4: Implement factual evidence and test it independently.**

```python
def render_evidence(*, lyric_used: bool, audio_state: str, top_label: str) -> str:
    fragments = [f"歌词信息支持“{top_label}”判断" if lyric_used else "歌词信息缺失"]
    if audio_state == "measured": fragments.append("音频已完成可用性与声学测量")
    if audio_state == "unavailable": fragments.append("音频不可用，未将其作为判定证据")
    return "；".join(fragments)[:500]
```

- [ ] **Step 5: Run the targeted tests and full suite.**

Run: `py -3.12 -m unittest discover -s competition_service/tests -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit.**

```powershell
git add competition_service/src/competition_emotion/service.py competition_service/src/competition_emotion/evidence.py competition_service/tests/test_service.py competition_service/tests/test_evidence.py
git commit -m "feat: implement official synchronous protocol"
```

## Task 3: Bounded audio request stage and graceful fallback

**Files:**
- Create: `competition_service/src/competition_emotion/request_audio.py`
- Create: `competition_service/tests/test_request_audio.py`
- Modify: `competition_service/src/competition_emotion/service.py`

- [ ] **Step 1: Write failing deadline and cleanup tests.**

```python
def test_request_audio_returns_unavailable_when_download_fails_and_deletes_temp_file(self) -> None:
    result = acquire_request_audio("https://audio.example/a.mp3", self.config)
    self.assertEqual(result.state, "unavailable")
    self.assertFalse(any(self.temp_dir.iterdir()))


def test_request_audio_never_exceeds_remaining_budget(self) -> None:
    result = acquire_request_audio("https://audio.example/a.mp3", RequestAudioConfig(total_seconds=0.01))
    self.assertEqual(result.state, "unavailable")
```

- [ ] **Step 2: Run the new tests and verify failure because `acquire_request_audio` does not exist.**

Run: `py -3.12 -m unittest competition_service.tests.test_request_audio -v`  
Expected: import failure naming the missing module.

- [ ] **Step 3: Implement request-scoped audio acquisition.**

```python
@dataclass(frozen=True)
class RequestAudioResult:
    state: Literal["measured", "unavailable"]
    features: dict[str, float] | None


def acquire_request_audio(url: str, config: RequestAudioConfig) -> RequestAudioResult:
    with TemporaryDirectory(dir=config.temp_root) as directory:
        try:
            download_audio(url, Path(directory) / "input.bin", max_bytes=config.max_bytes, timeout_seconds=remaining())
            waveform, sample_rate = decode_audio(Path(directory) / "input.bin", max_seconds=config.max_seconds)
            return RequestAudioResult("measured", measured_features(waveform, sample_rate))
        except (ValueError, RuntimeError, TimeoutError):
            return RequestAudioResult("unavailable", None)
```

Use a monotonic deadline and the existing explicit proxy/allowlist configuration. Never preserve raw audio or signed URLs. Integrate only the `measured/unavailable` fact into evidence; leave prediction score text-only until audio gain is proven.

- [ ] **Step 4: Run targeted and full tests.**

Run: `py -3.12 -m unittest discover -s competition_service/tests -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit.**

```powershell
git add competition_service/src/competition_emotion/request_audio.py competition_service/src/competition_emotion/service.py competition_service/tests/test_request_audio.py
git commit -m "feat: add bounded audio request fallback"
```

## Task 4: Retrain on the 2026-09-10 export and publish evaluation

**Files:**
- Modify: `competition_service/src/competition_emotion/train.py`
- Modify: `competition_service/tests/test_train.py`
- Create: `competition_service/scripts/train_official.ps1`
- Create: `docs/official-self-evaluation.md`

- [ ] **Step 1: Write failing report-provenance tests.**

```python
def test_training_report_records_formal_source_provenance(self) -> None:
    report = train_text_baseline(self.workbook, self.bundle_dir, labels=LABELS)
    self.assertEqual(report["source"]["file_name"], "emotion_songs_20260910.xlsx")
    self.assertIn("sha256", report["source"])
    self.assertIn("macro_recall", report["evaluation_metrics"])
```

- [ ] **Step 2: Run the targeted test and verify failure because report has no `source` object.**

Run: `py -3.12 -m unittest competition_service.tests.test_train.TrainTests.test_training_report_records_formal_source_provenance -v`  
Expected: assertion failure on missing `source`.

- [ ] **Step 3: Add immutable source provenance and training wrapper.**

```powershell
py -3.12 -m competition_emotion.train --workbook F:\netease\_music\competition\data\emotion_songs_20260910.xlsx --bundle-dir F:\netease\_music\competition\runs\official-20260910
```

The script must invoke this command, stop on error, then print the active bundle path and its report. The report must distinguish `strict_singleton_sample_count`, `multi_label_sample_count`, and all-song macro recall.

- [ ] **Step 4: Run the targeted tests, full suite, then the formal retrain command.**

Run: `py -3.12 -m unittest discover -s competition_service/tests -v`  
Expected: all tests pass.  
Run: `powershell -ExecutionPolicy Bypass -File competition_service/scripts/train_official.ps1`  
Expected: a new immutable bundle and self-evaluation report.

- [ ] **Step 5: Commit code, test, script and Markdown only.**

```powershell
git add competition_service/src/competition_emotion/train.py competition_service/tests/test_train.py competition_service/scripts/train_official.ps1 docs/official-self-evaluation.md
git commit -m "feat: retrain from updated official export"
```

## Task 5: Deployable internal-service operations and required documents

**Files:**
- Create: `competition_service/scripts/start_service.ps1`
- Create: `competition_service/scripts/restart_service.ps1`
- Create: `competition_service/service.env.example`
- Create: `docs/official-protocol-runbook.md`
- Create: `docs/official-cost-estimate.md`
- Modify: `README.md`
- Create: `competition_service/tests/test_scripts.py`

- [ ] **Step 1: Write failing script-content and default-binding tests.**

```python
def test_start_script_requires_non_loopback_host_and_bundle_root(self) -> None:
    content = START_SCRIPT.read_text(encoding="utf-8")
    self.assertIn("--host $BindHost", content)
    self.assertNotIn('default="127.0.0.1"', content)


def test_env_example_lists_audio_proxy_as_explicit_optional_setting(self) -> None:
    self.assertIn("AUDIO_PROXY_URL=", ENV_EXAMPLE.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run these tests and verify they fail because the scripts and environment template do not exist.**

Run: `py -3.12 -m unittest competition_service.tests.test_scripts -v`  
Expected: file-not-found failure.

- [ ] **Step 3: Implement repeatable operations.**

`start_service.ps1` accepts mandatory `-BundleRoot` and `-BindHost`, rejects loopback addresses, validates `-Port`, loads explicitly named environment settings, and starts `py -3.12 -m competition_emotion.service`. `restart_service.ps1` stops only the PID recorded in its service-state file, then invokes start. The runbook documents health check, logs, P50/P99 capture, audio failure recovery, model loading error, proxy allowlist mismatch, port conflicts, and process restart.

- [ ] **Step 4: Add a cost worksheet with formulas and explicit assumptions.**

The document must state measured P50/P99 after an end-to-end local sample run; CPU core-seconds and optional external-model cost are separate rows. It must not claim a cost value before a measured sample exists.

- [ ] **Step 5: Run all tests and a one-request Uvicorn smoke bound to a non-loopback private test address only when available; otherwise record the local-network limitation.**

Run: `py -3.12 -m unittest discover -s competition_service/tests -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit.**

```powershell
git add competition_service/scripts competition_service/service.env.example competition_service/tests/test_scripts.py docs/official-protocol-runbook.md docs/official-cost-estimate.md README.md
git commit -m "docs: add official deployment and cost materials"
```

## Plan verification

- Task 1 covers new export provenance and all three lyric sources.
- Task 2 covers the exact synchronous request/response protocol, Top-2, evidence, cost and error envelope.
- Task 3 covers mandatory audio URL handling, 25-second internal deadline and audio failure fallback.
- Task 4 covers retraining, song-level isolation and required self-evaluation metrics.
- Task 5 covers non-loopback deployment, restart, health, cost, monitoring and FAQ documentation.

No task writes raw audio, signed URLs or credentials to Git or logs. No task asserts an unmeasured 80% macro-recall result.
