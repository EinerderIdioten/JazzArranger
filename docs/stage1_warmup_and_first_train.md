# Stage 1 Warmup And First Training Step

This document records the current pre-training warmup state and the exact first
supervised training step to run later. It is a plan and runbook only; no training
is started by this document.

## Current Warmup State

Remote GPU machine:

```text
/root/autodl-tmp/jazzarranger
/root/autodl-tmp/models/qwen3-1.7b-base
```

Verified runtime:

```text
GPU: NVIDIA GeForce RTX 4090
CUDA available: true
torch: 2.5.1+cu124
transformers: 5.14.1
accelerate: 1.14.0
tokenizers: 0.22.2
safetensors: 0.8.0
```

Verified model files under `/root/autodl-tmp/models/qwen3-1.7b-base`:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

Warmup validation already performed:

```text
processed rows: 2075
datasets: EMOPIA+ 879, HLSD 11, POP909 1085, OpenBook 100
dataset weights: 1.0 for EMOPIA+/HLSD, 0.7 for POP909, 0.0 for OpenBook
training keys: C:maj 1323, C:min 752
original keys preserved: 24 distinct source keys
harmony token parse errors: 0
new harmony tokens: 55
duplicate new tokens: 0
```

Tokenizer/model dry run:

```text
base tokenizer length: 151669
added harmony tokens: 55
final tokenizer length: 151724
model input embedding rows before resize check: 151936
model LM head rows before resize check: 151936
model input embedding rows after resize check: 151936
model LM head rows after resize check: 151936
short forward pass: OK
observed warmup CUDA memory: about 3.23 GB
```

Tokenizer length dry check after C normalization:

```text
max_length 2048: train kept 925, skipped 672 over length
max_length 3072: train kept 1213, skipped 384 over length
max_length 4096: train kept 1513, skipped 84 over length
max_length 6144: train kept 1594, skipped 3 over length
```

The first attempted full-parameter run should therefore use `4096` as the
initial sequence length. If 24 GB memory is too tight with AdamW, fall back to
`3072` or add chunking instead of silently training on the heavily truncated
`2048` subset.

Important detail: this Qwen checkpoint has padded vocabulary matrices that are
larger than the tokenizer length. The resize helper therefore only grows the
model if the tokenizer becomes larger than the model matrices. It must not
shrink the model down to `len(tokenizer)`, because that would discard reserved
rows already present in the checkpoint.

## Stage Objective

The first training step is supervised sequence generation:

```text
standardized ABC melody -> tokenized harmony span sequence
```

The model should learn to generate a parseable, ordered harmony span sequence
that covers the melody grid:

```text
<HARMONY>
<SPAN> @0-4 <R_C> <Q_MAJ7> </SPAN>
<SPAN> @4-8 <R_G> <Q_DOM7> </SPAN>
</HARMONY>
```

This first step trains only the sequence objective. It does not train LoRA, RL,
DPO, GRPO, a note-role head, a change-point head, or a separate root/quality
classification head.

## Data Task

Training reads `data/processed/*_train.jsonl`. Validation reads
`data/processed/*_val.jsonl`.

Each JSONL row contains:

```text
abc_melody
harmony_tokens
dataset_weight
total_grid
dataset
split
normalization
original
```

The training fields are C-centered. `abc_melody`, `key`, `chords`, `harmony`,
and `harmony_tokens` have already been transposed so the source tonic is C.
The original source-key version is preserved in `original`, and
`normalization.transpose_semitones` records the reversible shift needed to map
generated roots back to the source key.

The prompt is built from the ABC only:

```text
<MELODY>
... ABC melody ...
</MELODY>
```

The target is `harmony_tokens`:

```text
<HARMONY>
<SPAN> @start-end <R_...> <Q_...> </SPAN>
</HARMONY>
```

OpenBook rows currently have `dataset_weight=0.0` and are skipped by default.
They are kept for audit, not for the first supervised loss, unless
`--include-zero-weight` is explicitly passed later.

## Masking

The training example concatenates:

```text
prompt_ids + target_ids + eos
```

The label mask is:

```text
prompt labels: -100
target labels: target token ids
padding labels: -100
```

This means the model receives the ABC melody as context, but loss is only charged
on the harmony output region. The model is not trained to reconstruct the ABC
prompt.

The attention mask is separate:

```text
real tokens: 1
padding tokens: 0
```

So there are two masks:

```text
attention_mask controls which tokens are visible to attention.
labels == -100 controls which tokens contribute to loss.
```

Dataset weighting is applied per example after token losses are averaged within
that example. With default filtering, the active sources are:

```text
EMOPIA+: weight 1.0
HLSD: weight 1.0
POP909: weight 0.7
OpenBook: excluded because weight is 0.0
```

## Loss

The first step uses weighted causal language modeling loss over target tokens:

```text
L_stage1_step1 = L_seq
```

For each batch, logits and labels are shifted by one token in the normal causal
LM way. Cross entropy is computed token by token with `ignore_index=-100`, so
prompt and padding positions do not contribute.

Per-example loss:

```text
loss_j = mean CE over unmasked target tokens in example j
```

Weighted batch loss:

```text
L_batch = sum_j weight_j * loss_j / sum_j weight_j
```

The first step does not include:

```text
L_pair
L_struct as a differentiable penalty
progression contrastive loss
preference loss
```

Structure is checked through parsing and validation during evaluation. If needed
later, constrained decoding or post-parse validation should be added before
turning structure into a training penalty.

## Model Changes

The base model remains a causal LM.

Changed:

```text
55 regular tokenizer tokens are added.
tokenizer pad token is set to eos if missing.
embedding/LM-head matrices are resized only if the tokenizer exceeds model rows.
the added token rows are trainable during full-parameter SFT.
```

Not changed:

```text
no Transformer block is added
no attention layer is added
no MLP layer is added
no LoRA adapter is added
no note-role head is added
no change-point head is added
no root/quality classifier head is added in step 1
```

The tokenizer with added tokens must be saved together with any trained
checkpoint. Otherwise the generated token ids for `<R_...>`, `<Q_...>`, and
`<SPAN>` will not be recoverable correctly.

## First Training Hyperparameters

Default first run for the 24 GB RTX 4090:

```text
model: /root/autodl-tmp/models/qwen3-1.7b-base
data_dir: /root/autodl-tmp/jazzarranger/data/processed
output_dir: /root/autodl-tmp/jazzarranger/outputs/stage1-qwen3-1.7b-base-full
max_length: 4096
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
effective batch size: 16 examples
num_train_epochs: 2
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.03
optimizer: adamw_torch
precision: bf16
gradient_checkpointing: true
logging_steps: 10
eval_steps: 100
save_steps: 100
save_total_limit: 2
report_to: none
```

Here `warmup_ratio=0.03` is optimizer learning-rate warmup. It is separate from
the environment/model warmup checks listed above.

Command to run later, only after explicit approval:

```bash
cd /root/autodl-tmp/jazzarranger
PYTHONPATH=/root/autodl-tmp/jazzarranger /root/miniconda3/bin/python3 -m src.train.sft_stage1 \
  --model-name-or-path /root/autodl-tmp/models/qwen3-1.7b-base \
  --data-dir data/processed \
  --output-dir outputs/stage1-qwen3-1.7b-base-full \
  --max-length 4096 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --save-total-limit 2 \
  --bf16 \
  --gradient-checkpointing
```

## First-Step Success Criteria

After this first training step, evaluate on validation data:

```text
parse success rate
span coverage accuracy
overlap/gap rate
root accuracy on gold boundaries
quality accuracy on gold boundaries
root+quality joint accuracy on gold boundaries
average generated chord count vs. gold chord count
```

The immediate priority is parseability and correct span coverage. Root/quality
accuracy matters, but stronger root-quality discrimination is expected to improve
more clearly after adding the later pairwise span scoring objective.
