# Official model self-evaluation

Run the formal baseline from the repository checkout:

```powershell
& .\competition_service\scripts\train_official.ps1 `
  -PlaylistPrior 'F:\netease\_music\competition\research_cache\netease_playlists\playlists_hot_500.json' `
  -FitAllForServing
```

The command publishes an immutable versioned bundle under
`F:\netease\_music\competition\runs\official-20260910`. Read `current.json`
to find `active_bundle`; the evaluation artifact is
`<BundleRoot>\<active_bundle>\report.json`.

The report is the source of record for the following historical baseline metrics.
The fixed holdout has since been reused in model and routing comparisons, so
it is development-exposed rather than an untouched blind test. See
`docs/2026-09-16-evaluation-audit.md` before interpreting these numbers.

| Measure | Report field | Definition |
| --- | --- | --- |
| Top-1 | `evaluation_metrics.strict_top1_accuracy` | Exact predicted-label accuracy on the `strict_singleton_sample_count` test songs that have exactly one positive official label. |
| Top-2 hit | `evaluation_metrics.strict_singleton_top2_hit_rate` / `any_positive_top2_hit_rate` | Whether the gold label appears in the first two candidates; this is the coverage available to the semantic review stage. |
| Top-7 hit | `evaluation_metrics.strict_singleton_top7_hit_rate` / `any_positive_top7_hit_rate` | Candidate coverage available to the semantic review stage. |
| Top-10 hit | `evaluation_metrics.strict_singleton_top10_hit_rate` / `any_positive_top10_hit_rate` | Maximum candidate-pool coverage before the adaptive semantic review stage. |
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

Measured from the formal run on 2026-09-14:

- Current report resolver: `F:\netease\_music\competition\runs\official-20260910\current.json` → `active_bundle` → `<BundleRoot>\<active_bundle>\report.json`
- The active bundle is resolved from `current.json`; the immutable bundle ID may change on each accepted retraining run.
- The active report is `<BundleRoot>\<active_bundle>\report.json`.
- Official source file: `emotion_songs_20260910.xlsx`
- Official source SHA-256: `18591837030e8d3005936dba6f43c7119cb9579aeaab8aafdf763acf379fe1af`
- Official source raw rows: 5,894
- Deduplicated official songs: 5,043 (4,034 official train; 1,009 official test; 0 split overlap)
- Training augmentation: none in the accepted v9 run. The earlier 54-row augmentation was retained as a research candidate but lowered the fixed holdout score.
- Model configuration: character TF-IDF (`char_wb`, n-grams 1–5, 150,000 features) with title/artist/album/song ID metadata repeated five times and `OneVsRest(LinearSVC(C=0.3, class_weight=None))`; SVC margins are converted to a row softmax for API confidence fields. Versioned high-agreement artist and album overrides are applied to final ranking, with artist evidence taking precedence. A public 12-category playlist snapshot is fitted as a weak `OneVsRest(LogisticRegression(C=0.1, class_weight="balanced"))` song-ID prior and fused with alpha 0.1; missing IDs use an all-zero playlist vector.
- Model version: `metadata-svc-v9`
- Serving artifact: after model selection, the API model is retrained on all 5,043 official songs; the fixed holdout metrics below remain computed from the disjoint 4,034-song fit split and are not recomputed after this serving retrain.
- Top-1: 0.5990016638935108 (601 strict-singleton test songs)
- Macro recall: 0.5458672967110856
- Any-positive Top-1: 0.6065411298315163
- Strict-singleton Top-2 coverage: 0.7603993344425957
- Strict-singleton Top-3 coverage: 0.8435940099833611
- Strict-singleton Top-7 coverage: 0.9534109816971714
- Strict-singleton Top-10 coverage: 0.9833610648918469
- Any-positive Top-2 coverage: 0.7750247770069376
- Any-positive Top-3 coverage: 0.8701684836471755
- Any-positive Top-7 coverage: 0.9672943508424182
- Any-positive Top-10 coverage: 0.9900891972249752
- Evaluation sample counts: 1,009 any-positive; 601 strict-singleton; 408 multi-label

The earlier v4 report used the workbook's first-level genre field, which is absent from the official request protocol, so it is retained only as an offline oracle and is not comparable to this protocol-compliant v9 score. The v9 local result does not meet the 95% final-accuracy target or the 80% macro-recall gate by itself. The production path therefore keeps candidate-limited semantic review and the human hard-case queue available for low-margin requests. A result on a newly frozen, previously uninspected labeled set is required before claiming the competition target.

On this fixed Test, the default `margin < 0.10` route sends 571 of the 601
strict-singleton songs to review. The adaptive pool uses 10 candidates for 387
hard cases, 7 for 109 cases, and 5 for 75 cases: 8.77 candidates per reviewed
song on average, about 12.3% fewer candidate slots than a fixed Top-10 pool.
The 30 directly released songs are 96.67% correct. This is only a routing
check; the actual reviewer must be evaluated end to end before claiming the
competition target. A reviewer that selects the correct candidate on at least
94.92% of routed singleton cases would clear 95% overall on this split; that is
a readiness calculation, not a measured reviewer result.

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
split remains unchanged, but historical comparisons have already exposed it
to development decisions. New thresholds must be selected inside Train/Dev;
the historical Test is only a regression check. Optional
`EMOTION_TRACE_PATH` writes append-only traces with model/Rubric versions,
candidate margin, reviewer outcome, and verified evidence. `mine_hard_cases`
and `evaluate_patch_gate` provide the review queue and regression gate for a
proposed Rubric patch; publishing is an explicit human-reviewed step.
