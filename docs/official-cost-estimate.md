# Official service cost estimate

This is a planning formula, not a measured production cost or performance claim. The current observed artifact on 2026-09-11 was the active `versions/3f46bab4daca4465a68098b5b8297c5b/model.joblib` under the official bundle pointer: 12,575,299 bytes (about 12.0 MiB). Its report declares model type `lyrics_tfidf_logreg`, model version `baseline-v1`, and report schema version 2.

| Cost component | Formula per recognition request | Current input/status |
| --- | --- | --- |
| CPU | CPU core seconds × internal rate | Measure CPU core seconds in the target host; internal rate is finance-owned. |
| Model storage | 12,575,299 bytes / bytes-per-GiB × storage rate × retention fraction | Current observed model artifact size; storage rate is environment-specific. |
| Audio transfer/decode | downloaded GiB × network rate + CPU core seconds × internal rate | Optional; record only when audio measurement is enabled. |
| External LLM tokens | input tokens × input-token rate + output tokens × output-token rate | not enabled / 0 |
| Total | sum of the rows above | Unknown until measured in the target environment. |

The planning target is `<=0.1` currency units per recognition request. This document does not claim that the target is met: CPU, storage, network, and operational rates still require operator measurements and internal rates.

P50 latency is unavailable until an end-to-end benchmark is measured on the deployment host with representative concurrent requests. P99 latency is unavailable for the same reason. Do not substitute a model load time, a unit test time, or a single local request for either percentile.

## Formal accuracy provenance

The exact formal source is [`official-self-evaluation.md`](official-self-evaluation.md), which identifies the active `report.json`. That report's source provenance is `emotion_songs_20260910.xlsx`, SHA-256 `18591837030e8d3005936dba6f43c7119cb9579aeaab8aafdf763acf379fe1af`, with 5,894 raw rows. For model version `baseline-v1`, its formal held-out metrics are `evaluation_metrics.strict_top1_accuracy = 0.5074875207986689` on 601 strict-singleton test songs, and `evaluation_metrics.macro_recall = 0.46969984911689106`. The report also records 1,009 any-positive held-out songs and 408 multi-label held-out songs.

These are offline baseline evaluation metrics, not a production accuracy guarantee. The report does not publish end-to-end latency percentiles, QPS, or cost.
