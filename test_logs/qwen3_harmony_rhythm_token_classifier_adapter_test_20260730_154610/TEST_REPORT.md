# Qwen3 Harmony-Rhythm Token Classifier

## Summary

This run predicts chord-change timing by gathering LM logits at fixed mask-token label slots. It does not autoregressively generate a JSON array.

| Item | Value |
| --- | --- |
| Run type | `qwen3_harmony_rhythm_token_classifier_eval` |
| Model path | `/root/autodl-tmp/models/Qwen3-8B` |
| Adapter path | `/root/autodl-tmp/JazzArranger/test_logs/qwen3_harmony_rhythm_token_classifier_train_20260730_153034/adapter` |
| Canonical source | `data/openbook/canonical` |
| Examples | `188` |
| Eval split | `test` |
| Window size | `4 bars` |
| Slot layout | `target_tail` |
| Mask token | `[MASK]` |
| Structural alignment | `100.0%` |
| Candidate coverage | `99.2%` |
| Threshold | `0.5` |
| Loss | `0.4157` |

## Metrics

| Metric | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Strict candidate match | 0.848 | 0.804 | 0.825 | 0.892 |
| ±0.5 beat tolerant | 0.848 | 0.804 | 0.825 | n/a |

## Counts

| Count | Value |
| --- | ---: |
| Gold chord-change candidates | 1013 |
| Predicted chord-change candidates | 960 |
| Total candidates | 3209 |
| Strict TP | 814 |
| Strict FP | 146 |
| Strict FN | 199 |
| Strict TN | 2050 |
| Harmony changes in windows | 1021 |
| Harmony changes covered by candidates | 1013 |
| Harmony changes uncovered by candidates | 8 |

## Highest-Error Samples

| Case | Gold change steps | Predicted change steps | Probability preview |
| --- | --- | --- | --- |
| `what_a_diffrence_a_day_made.canonical:bars_21_24` | `[2, 8, 13, 19]` | `[1, 4, 7, 9, 11, 15, 18, 20]` | `[0.988, 0.011, 0.001, 0.531, 0.001, 0.101, 0.965, 0.014, 0.547, 0.076, 0.969, 0.0, 0.029, 0.001, 0.5, 0.002, 0.095, 0.914, 0.015, 0.5, 0.148]` |
| `what_a_diffrence_a_day_made.canonical:bars_17_20` | `[7, 13, 19]` | `[1, 4, 6, 9, 12, 15, 18]` | `[0.98, 0.001, 0.003, 0.516, 0.072, 0.961, 0.017, 0.001, 0.516, 0.002, 0.072, 0.961, 0.019, 0.001, 0.5, 0.001, 0.076, 0.879, 0.018, 0.453, 0.141]` |
| `what_a_diffrence_a_day_made.canonical:bars_1_4` | `[3, 9, 13, 19]` | `[1, 4, 7, 11, 14]` | `[0.988, 0.363, 0.04, 0.879, 0.0, 0.002, 0.5, 0.001, 0.072, 0.002, 0.965, 0.371, 0.06, 0.867, 0.017, 0.001, 0.484, 0.001, 0.101, 0.003]` |
| `what_a_diffrence_a_day_made.canonical:bars_9_12` | `[6, 9, 16]` | `[1, 5, 7, 11, 15, 18]` | `[0.992, 0.0, 0.04, 0.001, 0.547, 0.081, 0.969, 0.484, 0.053, 0.001, 0.969, 0.0, 0.029, 0.001, 0.523, 0.081, 0.003, 0.887, 0.477, 0.002, 0.113, 0.004]` |
| `what_a_diffrence_a_day_made.canonical:bars_5_8` | `[3, 9, 13]` | `[1, 4, 7, 10, 14]` | `[0.988, 0.363, 0.047, 0.898, 0.009, 0.0, 0.5, 0.0, 0.063, 0.961, 0.002, 0.445, 0.063, 0.887, 0.408, 0.107, 0.002]` |
| `what_a_diffrence_a_day_made.canonical:bars_13_16` | `[5, 14, 18]` | `[1, 6, 11, 15]` | `[0.988, 0.0, 0.05, 0.469, 0.085, 0.957, 0.469, 0.0, 0.06, 0.001, 0.961, 0.013, 0.469, 0.063, 0.887, 0.438, 0.001, 0.09, 0.003]` |
| `one_by_one.canonical:bars_1_4` | `[1, 2, 3, 7, 8, 9, 10, 11, 12, 16]` | `[1, 7, 10, 16]` | `[0.992, 0.008, 0.469, 0.001, 0.085, 0.002, 0.934, 0.453, 0.057, 0.953, 0.011, 0.453, 0.001, 0.076, 0.002, 0.875, 0.377, 0.148]` |
| `come_rain_or_come_shine.canonical:bars_13_16` | `[1, 2, 5, 10, 12, 13, 14, 16, 18, 19]` | `[1, 5, 7, 10, 16, 18]` | `[0.988, 0.377, 0.047, 0.001, 0.949, 0.012, 0.516, 0.072, 0.001, 0.969, 0.0, 0.026, 0.484, 0.067, 0.001, 0.918, 0.019, 0.5, 0.133]` |

## Files

- `metadata.json`: run configuration
- `split_manifest.json`: tune-level train/validation/test split
- `scores.json`: aggregate metrics
- `predictions.jsonl`: one row per evaluated window, with probabilities
- `train_metrics.jsonl`: one row per epoch when `--mode train` is used
