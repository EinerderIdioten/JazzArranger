# Qwen3 Harmony-Rhythm Token Classifier

## Summary

This run predicts chord-change timing by gathering LM logits at fixed mask-token label slots. It does not autoregressively generate a JSON array.

| Item | Value |
| --- | --- |
| Run type | `qwen3_harmony_rhythm_token_classifier_eval` |
| Model path | `/root/autodl-tmp/models/Qwen3-8B` |
| Adapter path | `None` |
| Canonical source | `data/openbook/canonical` |
| Examples | `188` |
| Eval split | `test` |
| Window size | `4 bars` |
| Slot layout | `target_tail` |
| Mask token | `[MASK]` |
| Structural alignment | `100.0%` |
| Candidate coverage | `99.2%` |
| Threshold | `0.5` |
| Loss | `2.2001` |

## Metrics

| Metric | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Strict candidate match | 0.316 | 1.000 | 0.480 | 0.316 |
| ±0.5 beat tolerant | 0.316 | 1.000 | 0.480 | n/a |

## Counts

| Count | Value |
| --- | ---: |
| Gold chord-change candidates | 1013 |
| Predicted chord-change candidates | 3209 |
| Total candidates | 3209 |
| Strict TP | 1013 |
| Strict FP | 2196 |
| Strict FN | 0 |
| Strict TN | 0 |
| Harmony changes in windows | 1021 |
| Harmony changes covered by candidates | 1013 |
| Harmony changes uncovered by candidates | 8 |

## Highest-Error Samples

| Case | Gold change steps | Predicted change steps | Probability preview |
| --- | --- | --- | --- |
| `they_all_laughed.canonical:bars_13_16` | `[1, 10, 17]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` | `[0.73, 0.969, 0.953, 0.945, 0.895, 0.934, 0.957, 0.906, 0.93, 0.945, 0.961, 0.969, 0.969, 0.957, 0.961, 0.957, 0.945, 0.984, 0.984, 0.973, 0.973, 0.969, 0.961, 0.957]` |
| `if_you_could_see_me_now.canonical:bars_25_28` | `[1, 7, 15, 22]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` | `[0.73, 0.977, 0.953, 0.926, 0.945, 0.953, 0.961, 0.961, 0.941, 0.957, 0.977, 0.969, 0.934, 0.953, 0.949, 0.957, 0.941, 0.945, 0.965, 0.945, 0.953, 0.945, 0.957, 0.926]` |
| `if_you_could_see_me_now.canonical:bars_1_4` | `[1, 9, 15, 23]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` | `[0.777, 0.973, 0.945, 0.93, 0.941, 0.918, 0.945, 0.816, 0.957, 0.961, 0.969, 0.949, 0.965, 0.957, 0.953, 0.945, 0.953, 0.961, 0.941, 0.926, 0.918, 0.887, 0.93, 0.957]` |
| `if_you_could_see_me_now.canonical:bars_9_12` | `[1, 9, 14, 22]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` | `[0.707, 0.969, 0.945, 0.867, 0.895, 0.879, 0.91, 0.898, 0.926, 0.957, 0.957, 0.957, 0.957, 0.949, 0.945, 0.977, 0.98, 0.961, 0.984, 0.977, 0.945, 0.941, 0.957, 0.973]` |
| `they_all_laughed.canonical:bars_53_56` | `[1, 15, 21, 24]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]` | `[0.68, 0.953, 0.934, 0.906, 0.926, 0.938, 0.945, 0.969, 0.969, 0.941, 0.957, 0.949, 0.957, 0.965, 0.957, 0.969, 0.984, 0.941, 0.953, 0.938, 0.934, 0.969, 0.973, 0.938]` |
| `they_all_laughed.canonical:bars_25_28` | `[1, 5, 9, 12, 15, 18, 21]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]` | `[0.68, 0.961, 0.941, 0.918, 0.941, 0.926, 0.934, 0.898, 0.941, 0.945, 0.969, 0.957, 0.895, 0.957, 0.957, 0.973, 0.98, 0.969, 0.945, 0.926, 0.914, 0.965, 0.945, 0.914]` |
| `blue_monk.canonical:bars_5_8` | `[1, 7, 13, 17, 21]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]` | `[0.594, 0.973, 0.926, 0.906, 0.914, 0.844, 0.906, 0.93, 0.941, 0.91, 0.941, 0.867, 0.926, 0.957, 0.938, 0.941, 0.934, 0.898, 0.926, 0.898, 0.934, 0.957, 0.957, 0.906]` |
| `they_all_laughed.canonical:bars_45_48` | `[1, 8, 12]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]` | `[0.707, 0.969, 0.945, 0.914, 0.934, 0.898, 0.953, 0.934, 0.969, 0.945, 0.941, 0.945, 0.961, 0.977, 0.973, 0.957, 0.949, 0.906, 0.941, 0.93, 0.91, 0.918, 0.859, 0.906]` |

## Files

- `metadata.json`: run configuration
- `split_manifest.json`: tune-level train/validation/test split
- `scores.json`: aggregate metrics
- `predictions.jsonl`: one row per evaluated window, with probabilities
- `train_metrics.jsonl`: one row per epoch when `--mode train` is used
