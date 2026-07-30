# Qwen3 Harmony-Rhythm Token Classifier

## Summary

This run predicts chord-change timing by gathering LM logits at fixed mask-token label slots. It does not autoregressively generate a JSON array.

| Item | Value |
| --- | --- |
| Run type | `qwen3_harmony_rhythm_token_classifier_train` |
| Model path | `/root/autodl-tmp/models/Qwen3-8B` |
| Adapter path | `None` |
| Canonical source | `data/openbook/canonical` |
| Examples | `270` |
| Eval split | `val` |
| Window size | `4 bars` |
| Slot layout | `target_tail` |
| Mask token | `[MASK]` |
| Structural alignment | `100.0%` |
| Candidate coverage | `99.0%` |
| Threshold | `0.5` |
| Loss | `0.3290` |

## Metrics

| Metric | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Strict candidate match | 0.764 | 0.846 | 0.803 | 0.896 |
| ±0.5 beat tolerant | 0.764 | 0.846 | 0.803 | n/a |

## Counts

| Count | Value |
| --- | ---: |
| Gold chord-change candidates | 1351 |
| Predicted chord-change candidates | 1496 |
| Total candidates | 5419 |
| Strict TP | 1143 |
| Strict FP | 353 |
| Strict FN | 208 |
| Strict TN | 3715 |
| Harmony changes in windows | 1364 |
| Harmony changes covered by candidates | 1351 |
| Harmony changes uncovered by candidates | 13 |

## Highest-Error Samples

| Case | Gold change steps | Predicted change steps | Probability preview |
| --- | --- | --- | --- |
| `nows_the_time.canonical:bars_1_4` | `[1]` | `[1, 5, 8, 12, 15, 19, 23, 27]` | `[0.992, 0.0, 0.031, 0.001, 0.581, 0.078, 0.002, 0.966, 0.0, 0.026, 0.001, 0.604, 0.077, 0.002, 0.968, 0.0, 0.028, 0.001, 0.526, 0.001, 0.091, 0.003, 0.911, 0.001]` |
| `ill_remember_april.canonical:bars_33_36` | `[2, 7, 9, 11, 17]` | `[1, 4, 9, 15]` | `[0.978, 0.393, 0.051, 0.894, 0.0, 0.001, 0.486, 0.069, 0.958, 0.015, 0.459, 0.001, 0.06, 0.004, 0.896, 0.02, 0.458, 0.139]` |
| `opus_one.canonical:bars_25_28` | `[1, 14]` | `[1, 3, 7, 10, 14, 17, 21, 24]` | `[0.982, 0.0, 0.566, 0.001, 0.086, 0.004, 0.973, 0.0, 0.001, 0.551, 0.002, 0.096, 0.003, 0.968, 0.022, 0.001, 0.524, 0.002, 0.093, 0.003, 0.929, 0.001, 0.057, 0.529]` |
| `robbins_nest.canonical:bars_1_4` | `[1, 16]` | `[1, 5, 8, 10, 16, 20, 22, 25]` | `[0.99, 0.0, 0.03, 0.001, 0.558, 0.001, 0.083, 0.959, 0.0, 0.561, 0.001, 0.01, 0.086, 0.001, 0.003, 0.97, 0.001, 0.028, 0.001, 0.535, 0.086, 0.908, 0.022, 0.001]` |
| `ill_remember_april.canonical:bars_37_40` | `[3, 10, 15, 18, 19]` | `[1, 7, 10, 12, 18]` | `[0.977, 0.016, 0.489, 0.001, 0.094, 0.003, 0.964, 0.0, 0.002, 0.512, 0.084, 0.965, 0.001, 0.002, 0.494, 0.001, 0.083, 0.896, 0.409, 0.127]` |
| `opus_one.canonical:bars_9_12` | `[1, 15]` | `[1, 5, 8, 11, 15, 17, 21, 24]` | `[0.989, 0.0, 0.048, 0.001, 0.557, 0.083, 0.002, 0.975, 0.0, 0.003, 0.559, 0.001, 0.087, 0.003, 0.971, 0.0, 0.515, 0.001, 0.069, 0.003, 0.929, 0.001, 0.002, 0.523]` |
| `opus_one.canonical:bars_1_4` | `[1, 14]` | `[1, 4, 7, 10, 14, 16, 20, 22]` | `[0.992, 0.0, 0.041, 0.542, 0.082, 0.001, 0.964, 0.0, 0.034, 0.573, 0.001, 0.079, 0.002, 0.97, 0.015, 0.5, 0.001, 0.086, 0.002, 0.903, 0.001, 0.523, 0.002, 0.131]` |
| `bessies_blues.canonical:bars_5_8` | `[1, 16]` | `[1, 4, 8, 12, 16, 19, 23, 27]` | `[0.99, 0.015, 0.001, 0.538, 0.001, 0.092, 0.003, 0.972, 0.0, 0.035, 0.001, 0.567, 0.001, 0.099, 0.003, 0.964, 0.019, 0.001, 0.522, 0.002, 0.094, 0.002, 0.924, 0.001]` |

## Files

- `metadata.json`: run configuration
- `split_manifest.json`: tune-level train/validation/test split
- `scores.json`: aggregate metrics
- `predictions.jsonl`: one row per evaluated window, with probabilities
- `train_metrics.jsonl`: one row per epoch when `--mode train` is used
