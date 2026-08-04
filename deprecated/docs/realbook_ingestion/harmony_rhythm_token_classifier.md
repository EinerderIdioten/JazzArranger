# Harmony-Rhythm Token Classifier

This document defines the Phase 1 fine-tuning path after the JSON-generation
zero-shot baseline. It keeps the elastic candidate renderer, but changes the
model objective from text generation to fixed-length sequence labeling.

## Pipeline Step

This is the model-training step after canonical storage has been rendered into
harmony-rhythm candidates:

```text
canonical lead-sheet storage
  -> elastic harmony-rhythm candidate rendering
  -> token-level label-slot rendering
  -> Qwen causal LM forward pass
  -> logits gathered at label-slot mask tokens
  -> binary loss and deterministic scoring
```

The model input still excludes chord names and `harmony_stream`. The
`harmony_stream` is used only to create labels and evaluation metrics.

## Why This Replaces JSON Generation

The zero-shot JSON baseline can fail because a decoder-only model must
autoregressively emit brackets, commas, and all `0`/`1` values. A single extra
or missing token causes the output length to drift.

The token-classifier route removes that failure mode. The renderer creates one
label slot for each candidate checkpoint and records the token index of each
slot marker. The model never has to generate an array. In training and
inference, the code gathers logits at those recorded positions and converts
them directly into binary labels.

Structural alignment is therefore controlled by code:

```text
number of candidate rows == number of mask-token positions == number of labels
```

If that invariant fails, the sample is rejected before training or inference.

## Causal-LM Layout

Qwen is a decoder-only causal LM. If the mask marker is placed inline inside
each candidate row, that position can only attend to earlier text and the
current row. It cannot see later melody candidates.

Default layout therefore uses a target-tail block:

```text
候选:
[[bar,beat,type,pitch,duration], ...]
标签位:
[[bar,beat,[MASK]], ...]
```

Each label slot appears after the full candidate table, so every gathered
hidden state can attend to the complete melody-candidate input. The `bar` and
`beat` copied into each label slot make the row-to-label alignment explicit.

The inline layout is still valid for strictly left-to-right experiments, but it
is not the default for Qwen.

## Mask Token

The mask marker must tokenize to exactly one token. The implementation uses
`[MASK]` by default. If the tokenizer does not already encode it as one token,
the script adds it as an additional special token and resizes the model
embeddings.

When a new `[MASK]` token is added, its embedding is initialized from the
existing `M` token when possible. This avoids starting from a completely random
input vector while preserving the user-facing placeholder semantics.

## Binary Logit

At each recorded mask-token position, the script reads the causal LM logits for
the existing one-token class anchors `0` and `1`.

The binary classifier logit is:

```text
logit_change = lm_logit("1") - lm_logit("0")
```

This allows `BCEWithLogitsLoss` to train the model without adding a separate
classification head.

## Loss

The training loss is:

```text
total_loss = BCE + alpha * sparsity + beta * smoothness
```

Terms:

- `BCE`: binary cross entropy on valid label slots only.
- `pos_weight`: increases the cost of missing real chord changes.
- `sparsity`: mean predicted probability over valid slots, discouraging the
  model from predicting too many chord changes.
- `smoothness`: mean product of adjacent probabilities, discouraging repeated
  high-probability `1,1` runs.

The implementation masks padding positions for all three terms. This matters
because different four-bar windows can have different candidate counts under
the elastic grid.

Default starting coefficients:

```text
pos_weight = 1.5
alpha = 0.1
beta = 0.3
```

These should be tuned against validation precision/recall. If the model still
over-predicts chord changes, raise `alpha` or `beta`, or raise the inference
threshold above `0.5`.

## Evaluation

Evaluation remains a binary sequence-labeling task:

- strict candidate-position precision, recall, F1, and accuracy
- optional tolerant F1 within `0.5` beat
- candidate coverage against `harmony_stream`
- structural alignment pass rate

The old format pass rate becomes a structural invariant rather than a model
generation metric. A valid token-classifier batch has fixed output length by
construction.

## Data Split

Split by tune, not by four-bar window. The split key is `source_file`, which
corresponds to one OpenBook tune in the current canonical snapshot.

Default split:

```text
train: 75% of tunes
val:   15% of tunes
test:  10% of tunes
seed:  20260730
```

Training mode uses `train` for optimization and `val` for epoch-level metrics.
Evaluation mode uses `test` by default. The script writes `split_manifest.json`
so the exact tune membership is auditable.

For robustness checks, use `--split-mode kfold --num-folds 5 --fold-index N`.
In k-fold mode the selected fold becomes `test`; validation is sampled from
the remaining tunes with `--val-fraction`, and the rest is training data. This
keeps the unit of separation at tune level while allowing every tune to serve
as a held-out test tune across a full fold sweep.

## Script

Use:

```bash
python src/realbook_ingestion/scripts/train_qwen3_harmony_rhythm_token_classifier.py \
  --canonical-dir data/openbook/canonical \
  --model-path /root/autodl-tmp/models/Qwen3-8B \
  --output-dir test_logs/qwen3_harmony_rhythm_token_classifier_smoke \
  --mode train \
  --use-lora \
  --load-in-4bit \
  --gradient-checkpointing \
  --eval-split auto \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --log-every-steps 50 \
  --save-best-artifacts
```

Initial GPU plan is single-card QLoRA. Do not introduce DeepSpeed until the
plain LoRA path is shown to be too slow or too memory-heavy.

Recommended experiment order:

1. Run one fixed-split 3-epoch training job with `--save-best-artifacts`.
2. Evaluate `best_adapter` on the same fixed test split.
3. Run a small multi-seed sweep, e.g. `20260730`, `20260731`, `20260732`, to
   estimate split sensitivity.
4. Run 5-fold CV only after the fixed-split run is stable enough to justify the
   GPU time.
