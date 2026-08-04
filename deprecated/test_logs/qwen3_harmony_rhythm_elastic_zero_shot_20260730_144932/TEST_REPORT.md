# Qwen3 Harmony-Rhythm Baseline

## Summary

This is a zero-shot, pre-finetuning baseline for predicting chord-change timing from elastic melody-checkpoint prompts. The prompt does not include chord names or gold harmony labels.

| Item | Value |
| --- | --- |
| Model path | `/root/autodl-tmp/models/Qwen3-8B` |
| Canonical source | `data/openbook/canonical` |
| Examples | `40` |
| Window size | `4 bars` |
| Candidate policy | `elastic_melody_backbone_v1` |
| Sampling seed | `20260730` |
| Format pass rate | `82.5%` |
| Candidate coverage | `97.8%` |

## Metrics

Primary metrics treat chord-change positions as label `1` over generated candidate checkpoints. Invalid model outputs are scored as all-zero predictions in `all_samples` metrics, so format failures are penalized.

| Metric | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Strict candidate match | 0.278 | 0.440 | 0.341 | 0.517 |
| ±0.5 beat tolerant | 0.342 | 0.541 | 0.419 | n/a |

## Counts

| Count | Value |
| --- | ---: |
| Gold chord-change candidates | 218 |
| Predicted chord-change candidates | 345 |
| Total candidates | 768 |
| Strict TP | 96 |
| Strict FP | 249 |
| Strict FN | 122 |
| Strict TN | 301 |
| Harmony changes in windows | 223 |
| Harmony changes covered by candidates | 218 |
| Harmony changes uncovered by candidates | 5 |

## Format Failures

- Passed: `33`
- Failed: `7`

## Highest-Error Samples

| Case | Format | Gold change steps | Predicted change steps | Raw output preview |
| --- | --- | --- | --- | --- |
| `my_one_and_only_love.canonical:bars_13_16` | `wrong_length:27` | `[1, 3, 6, 8, 9, 11, 15, 18, 19, 21, 24, 27]` | `[]` | [1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] |
| `the_things_we_did_last_summer.canonical:bars_21_24` | `wrong_length:33` | `[1, 5, 9, 13, 18, 23, 28]` | `[]` | [1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0] |
| `i_concentrate_on_you.canonical:bars_85_88` | `wrong_length:20` | `[1, 8, 10, 12, 16, 19]` | `[]` | [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] |
| `heres_that_rainy_day.canonical:bars_13_16` | `wrong_length:21` | `[1, 8, 13, 14, 17, 20]` | `[]` | [1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,0,0] |
| `desafinado.canonical:bars_53_56` | `wrong_length:20` | `[1, 8, 13, 14, 18]` | `[]` | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| `blue_skies.canonical:bars_5_8` | `wrong_length:14` | `[1, 4, 5, 9, 12]` | `[]` | [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1] |
| `stella_by_starlight.canonical:bars_13_16` | `wrong_length:17` | `[1, 7, 10, 12, 16]` | `[]` | [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] |
| `ill_remember_april.canonical:bars_25_28` | `ok` | `[1, 7, 12, 18]` | `[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]` | [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |

## Files

- `metadata.json`: run configuration
- `scores.json`: aggregate metrics
- `predictions.jsonl`: one row per evaluated window, with raw model output
- `sample_prompts.jsonl`: prompt messages for audit
