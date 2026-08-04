# Stage-One Training

Stage one is full-parameter supervised fine-tuning:

```text
ABC melody -> tokenized harmony spans
```

The model remains a causal LM. There is no LoRA, no RL/DPO, no note-role head,
and no change-point head.

## Tokenizer

The project adds 55 harmony tokens from `src.data.harmony_tokens.NEW_TOKENS`.
They are regular added tokens, not skipped special tokens. After adding them,
the model embedding matrix and LM head must be resized:

```bash
python3 -m src.train.tokenizer_setup \
  --model-name-or-path Qwen/Qwen3-1.7B-Base \
  --output-dir outputs/tokenizer-qwen3-1.7b-base \
  --trust-remote-code
```

The training script performs the same tokenizer setup automatically before
training.

## Data

Training uses `data/processed/*_train.jsonl`; validation uses
`data/processed/*_val.jsonl`.

The main processed fields are already key-normalized for training. `abc_melody`,
`key`, `chords`, `harmony`, and `harmony_tokens` are C-centered; original
pre-transposition values are kept under `original`, with the reversible shift in
`normalization.transpose_semitones`.

Rows with `dataset_weight <= 0` are excluded by default. That means OpenBook is
available for audit but not used in the first supervised run unless
`--include-zero-weight` is set.

The training target is the `harmony_tokens` field:

```text
<HARMONY>
<SPAN> @0-4 <R_C> <Q_MAJ7> </SPAN>
<SPAN> @4-8 <R_G> <Q_DOM7> </SPAN>
</HARMONY>
```

The prompt wraps ABC in melody markers:

```text
<MELODY>
... ABC melody ...
</MELODY>
```

## 4090 24GB First Run

Use conservative full-parameter settings first:

```bash
python3 -m src.train.sft_stage1 \
  --model-name-or-path Qwen/Qwen3-1.7B-Base \
  --data-dir data/processed \
  --output-dir outputs/stage1-qwen3-1.7b-base-full \
  --max-length 4096 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --bf16 \
  --gradient-checkpointing \
  --trust-remote-code
```

Post-normalization tokenizer dry checks showed that `--max-length 2048` skips too
many active training rows. `4096` keeps most rows while remaining the first
reasonable setting to try on a 24 GB 4090. If it is still too tight for
full-parameter AdamW, drop to `3072` or add a chunking preprocessing step before
training.

## Eval

```bash
python3 -m src.train.eval_stage1 \
  --model-name-or-path outputs/stage1-qwen3-1.7b-base-full \
  --data-dir data/processed \
  --split val \
  --output-dir outputs/stage1-eval \
  --trust-remote-code
```

The eval script reports parse success plus boundary/root/quality/joint accuracy
on gold span boundaries.
