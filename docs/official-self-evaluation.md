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
- Observed active bundle for this run: `versions/9a287f66ef8947ab9117900b187c5130`
- Observed report for this run: `F:\netease\_music\competition\runs\official-20260910\versions\9a287f66ef8947ab9117900b187c5130\report.json`
- Source file name: `emotion_songs_20260910.xlsx`
- Source SHA-256: `18591837030e8d3005936dba6f43c7119cb9579aeaab8aafdf763acf379fe1af`
- Source raw rows: 5,894
- Deduplicated songs: 5,043 (4,034 train; 1,009 test; 0 split overlap)
- Model configuration: character TF-IDF (`char_wb`, n-grams 1–5, 150,000 features) with title/artist/genre metadata repeated five times and balanced one-vs-rest `LinearSVC(C=0.3)`; SVC margins are converted to a row softmax for the API confidence fields.
- Top-1: 0.6089850249584027 (601 strict-singleton test songs)
- Macro recall: 0.5605640997855902
- Any-positive Top-1: 0.6184340931615461
- Strict-singleton Top-2 coverage: 0.7753743760399334
- Strict-singleton Top-3 coverage: 0.8552412645590682
- Any-positive Top-2 coverage: 0.7908820614469773
- Any-positive Top-3 coverage: 0.8731417244796829
- Evaluation sample counts: 1,009 any-positive; 601 strict-singleton; 408 multi-label

The model change is an offline improvement only. The macro-recall hard gate is
80%, so this local text-only model is not presented as meeting the competition
gate. The next production stage is a candidate-limited semantic review for
low-margin cases; its endpoint and measured results must be supplied before
claiming a gate pass.
