# Official model self-evaluation

Run the formal baseline from the repository checkout:

```powershell
& .\competition_service\scripts\train_official.ps1
```

The command publishes an immutable versioned bundle under
`F:\netease\_music\competition\runs\official-20260910`. Read `current.json`
to find `active_bundle`; the evaluation artifact is
`<BundleRoot>\<active_bundle>\report.json`.

The report is the source of record for the following baseline metrics:

| Measure | Report field | Definition |
| --- | --- | --- |
| Top-1 | `evaluation_metrics.strict_top1_accuracy` | Exact predicted-label accuracy on the `strict_singleton_sample_count` test songs that have exactly one positive official label. |
| Top-2 hit | `evaluation_metrics.strict_singleton_top2_hit_rate` / `any_positive_top2_hit_rate` | Whether the gold label appears in the first two candidates; this is the coverage available to the semantic review stage. |
| Top-7 hit | `evaluation_metrics.strict_singleton_top7_hit_rate` / `any_positive_top7_hit_rate` | Candidate coverage available to the semantic review stage. |
| Top-10 hit | `evaluation_metrics.strict_singleton_top10_hit_rate` / `any_positive_top10_hit_rate` | Higher-coverage candidate pool used by the default semantic review stage. |
| Macro recall | `evaluation_metrics.macro_recall` | Mean per-label recall across the configured official labels on the held-out test set. |

`evaluation_sample_counts` supplies the denominators. `any_positive_sample_count`
counts all held-out songs with at least one positive label;
`strict_singleton_sample_count` is the Top-1 denominator; and
`multi_label_sample_count` counts held-out songs with more than one positive
label. `counts.total_songs`, `counts.train_songs`, and `counts.test_songs`
refer to deduplicated song IDs, while `source.rows` is the raw workbook-row
count before validity filtering or song-ID grouping.

Every accepted report must have schema version 2 and a `source` object with
exactly `file_name`, `sha256`, and `rows`. Record measured values only from the
active `report.json`; do not infer them from console output or use a target
percentage as an evaluation result.

## Latest formal run

Measured from the formal run on 2026-09-11:

- Current report resolver: `F:\netease\_music\competition\runs\official-20260910\current.json` → `active_bundle` → `<BundleRoot>\<active_bundle>\report.json`
- Observed active bundle for this run: `versions/132db19a12c14cf1a18a5da0d84b634b`
- Observed report for this run: `F:\netease\_music\competition\runs\official-20260910\versions\132db19a12c14cf1a18a5da0d84b634b\report.json`
- Official source file: `emotion_songs_20260910.xlsx`
- Official source SHA-256: `18591837030e8d3005936dba6f43c7119cb9579aeaab8aafdf763acf379fe1af`
- Official source raw rows: 5,894
- Deduplicated official songs: 5,043 (4,034 official train; 1,009 official test; 0 split overlap)
- Training augmentation: 54 song IDs from `emotion_songs_20260908_with_lrc.xlsx`; all 5,043 overlapping IDs were excluded, so no official test ID was added to training.
- Model configuration: character TF-IDF (`char_wb`, n-grams 1–5, 150,000 features) with title/artist/album metadata repeated five times and `OneVsRest(LinearSVC(C=0.3, class_weight=None))`; SVC margins are converted to a row softmax for API confidence fields. A versioned high-agreement artist override is applied to final ranking.
- Top-1: 0.5773710482529119 (601 strict-singleton test songs)
- Macro recall: 0.5257368444999785
- Any-positive Top-1: 0.5877106045589693
- Strict-singleton Top-2 coverage: 0.7487520798668885
- Strict-singleton Top-3 coverage: 0.8186356073211315
- Strict-singleton Top-7 coverage: 0.9500831946755408
- Strict-singleton Top-10 coverage: 0.978369384359401
- Any-positive Top-2 coverage: 0.7641228939544104
- Any-positive Top-3 coverage: 0.8374628344895937
- Any-positive Top-7 coverage: 0.9663032705649157
- Any-positive Top-10 coverage: 0.9871159563924677
- Evaluation sample counts: 1,009 any-positive; 601 strict-singleton; 408 multi-label

The earlier v4 report used the workbook's first-level genre field, which is absent from the official request protocol, so it is retained only as an offline oracle and is not comparable to this protocol-compliant v5 score. The v5 local result does not meet the 95% final-accuracy target or the 80% macro-recall gate by itself. The production path therefore keeps the candidate-limited internal semantic reviewer enabled for low-margin requests. Its independent Dev/Test result must be recorded before claiming the competition target.

On this fixed Test, the default `margin < 0.10` route sends 567 of the 601
strict-singleton songs to review; their correct label is present in the Top-10
pool for 97.71% of routed songs, while the 34 directly released songs are
97.06% correct. This is only a candidate-pool recall check. Because there are
15 labels, an uninformative pool of 10 labels would already cover about 66.7%;
97.71% must not be reported as 97% accuracy. Conditionally, a reviewer that
selects the correct candidate on at least 95.3% of routed singleton cases would
clear 95% overall on this split. That is a readiness calculation, not a
measured reviewer result; the actual endpoint must be evaluated end to end.

## Rubric, evidence, and calibration controls

The repository now includes a versioned 15-label Rubric at
`competition_service/rubric/emotion_rubric.json`. The semantic review payload
contains only the local candidate labels and their Rubric context. The Rubric is
soft context: it does not hard-exclude reasonable same-polarity labels, which
matches the competition rule that the final output is one most-confident
second-level label.

Reviewer evidence is accepted only when every returned lyric quote is found in
the request lyrics and every returned rule ID exists in the active Rubric. An
invalid or unavailable reviewer result falls back to the local model.

`python -m competition_emotion.calibration` creates a deterministic Dev slice
inside Official Train (`20260911`, 15% by default). The fixed Official Test
split remains unchanged and is never used for threshold tuning. Optional
`EMOTION_TRACE_PATH` writes append-only traces with model/Rubric versions,
candidate margin, reviewer outcome, and verified evidence. `mine_hard_cases`
and `evaluate_patch_gate` provide the review queue and regression gate for a
proposed Rubric patch; publishing is an explicit human-reviewed step.
