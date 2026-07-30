# Qwen3 Harmony-Rhythm Token Classifier Comparison

This report compares the base Qwen3-8B token-classifier logits against the one-epoch QLoRA adapter on the same tune-level test split.

## Split

| Split | Tunes | Windows |
| --- | ---: | ---: |
| train | 116 | 1169 |
| val | 23 | 270 |
| test | 16 | 188 |

Default split is tune-level by `source_file` with seed `20260730`; no tune appears in more than one split.

## Metrics

| Run | Examples | Precision | Recall | F1 | Accuracy | Gold + rate | Pred + rate | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base test, no training | 188 | 0.316 | 1.000 | 0.480 | 0.316 | 0.316 | 1.000 | 2.200 |
| Trained adapter validation | 270 | 0.764 | 0.846 | 0.803 | 0.896 | 0.249 | 0.276 | 0.329 |
| Trained adapter test | 188 | 0.848 | 0.804 | 0.825 | 0.892 | 0.316 | 0.299 | 0.416 |

## Counts

| Run | TP | FP | FN | TN | Gold + | Pred + | Total candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base test, no training | 1013 | 2196 | 0 | 0 | 1013 | 3209 | 3209 |
| Trained adapter validation | 1143 | 353 | 208 | 3715 | 1351 | 1496 | 5419 |
| Trained adapter test | 814 | 146 | 199 | 2050 | 1013 | 960 | 3209 |

## Interpretation

- The base model is structurally aligned but predicts every candidate as a chord change on the test set.
- One epoch of QLoRA changes the task from all-positive behavior to a calibrated sequence-labeling classifier.
- Test precision rises sharply while recall remains usable; the predicted positive rate becomes close to the gold positive rate.
- Candidate coverage on the test split is unchanged because it is determined by the renderer, not the model.

## Artifacts

- Base test log: `/root/autodl-tmp/JazzArranger/test_logs/qwen3_harmony_rhythm_token_classifier_base_test_20260730_152836`
- Training log and adapter: `/root/autodl-tmp/JazzArranger/test_logs/qwen3_harmony_rhythm_token_classifier_train_20260730_153034`
- Adapter test log: `/root/autodl-tmp/JazzArranger/test_logs/qwen3_harmony_rhythm_token_classifier_adapter_test_20260730_154610`
- Adapter path: `/root/autodl-tmp/JazzArranger/test_logs/qwen3_harmony_rhythm_token_classifier_train_20260730_153034/adapter`
