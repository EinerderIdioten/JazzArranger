# Chord-Aware Qwen Tokenizer

This document defines the first chord-aware tokenizer layer for Qwen2.5
experiments. It does not replace Qwen's original vocabulary. It appends a
small closed set of special tokens that describe Real Book-level chord
structure.

## Storage Boundary

Canonical storage remains the source of truth and keeps raw chord symbols in
`harmony_stream`:

```json
[bar, beat, "G7b9", duration_beats]
```

The chord-aware representation is a derived training layer. It parses the raw
symbol into:

```json
{
  "surface": "G7b9",
  "root": {"spelling": "G", "pc": 7},
  "quality": "7",
  "tensions": ["b9"],
  "bass": null
}
```

This lets later training use structured root/quality labels without losing the
original score spelling.

## Token Design

The active first-stage scope is:

```text
Root + Quality
```

The schema also reserves room for:

```text
Tensions + Bass + Surface chord tokens
```

The default token list includes:

- control tokens: `<HOLD>`, `<NEW>`, `<CHANGE:0>`, `<CHANGE:1>`
- chord wrappers: `<CHORD>`, `</CHORD>`
- root tokens: `<ROOT:C>`, `<ROOT:Db>`, ... plus enharmonic spellings
- quality tokens: `<QUAL:maj>`, `<QUAL:m7>`, `<QUAL:7>`, `<QUAL:maj7>`, ...
- tension tokens: `<TENS:b9>`, `<TENS:9>`, `<TENS:#11>`, `<TENS:alt>`, ...
- future bass tokens: `<BASS:C>`, `<BASS:Db>`, ...

The tokenizer extension intentionally does not add one token per observed
chord by default. Surface chord tokens can be added later with
`--include-surface-tokens`, but the first-stage path favors factorized
generalization.

## Pitch Class

Each root keeps both spelling and pitch class:

```json
{"spelling": "Db", "pc": 1}
{"spelling": "C#", "pc": 1}
```

`pc` is the 12-tone pitch class used for interval and circle-of-fifths
features. Spelling is kept because enharmonic equivalents can have different
musical meaning and should be recoverable in output.

The manifest also stores sinusoidal pitch-class and circle-of-fifths features.
Those features are not injected into Qwen by the tokenizer itself; they are
available for a later embedding-initialization or projector step.

## Embedding Initialization

New special tokens do not inherit Qwen's pretrained musical meaning by simply
being added to the tokenizer. Before continued pretraining or full-parameter
SFT, initialize their input embeddings and tied LM-head rows from nearby
existing Qwen tokens.

Starting policy:

- root tokens: initialize `<ROOT:D>` from the existing `D` / space-prefixed
  `D` token embedding when available.
- bass tokens: use the same initializer as the matching root token.
- quality tokens: initialize from text fragments such as `m7`, `maj7`, `7`,
  `dim`, `sus`, or from a short average of component tokens when the fragment
  is split by the base tokenizer.
- tension tokens: initialize from fragments such as `b9`, `#11`, `13`, `alt`,
  or from an average of their component token embeddings.
- control/wrapper tokens: initialize from short textual anchors such as `new`,
  `hold`, `chord`, `root`, `quality`, and `tension`.

After initialization, run a short embedding warmup before full training:

```text
freeze transformer blocks
train resized input embedding / LM head on chord-DSL text
then unfreeze for full-parameter continued pretraining or SFT
```

The current OpenBook snapshot has about ten thousand harmony events, which is
enough to validate the pipeline but too small for stable harmonic
generalization. Transposition augmentation across 12 keys should be used before
serious training; it turns the same tune-level corpus into roughly twelve
times as many root/quality events while preserving functional patterns.

## Scope-1 Mask Training Loss

The first full-parameter training scope predicts chord-change timing plus the
basic chord root and quality at every rendered candidate checkpoint. Each
candidate contributes three target-tail mask positions:

```text
[bar, beat, M, M, M]
           |  |  |
           |  |  quality mask
           |  root mask
           change mask
```

The model is still a causal LM. The trainer gathers logits only at these mask
positions instead of asking the model to freely generate an answer string.

Notation:

- `m[b,t]`: valid slot mask, excluding padding.
- `y[b,t]`: gold change label, `1` if `harmony_stream` starts a new chord at
  slot `t`, otherwise `0`.
- `c0[b,t]`, `c1[b,t]`: LM logits for `<CHANGE:0>` and `<CHANGE:1>` at the
  change mask.
- `s[b,t] = c1[b,t] - c0[b,t]`: binary change logit.
- `p[b,t] = sigmoid(s[b,t])`: predicted change probability.
- `r[b,t]`: gold root class index, valid only when `y[b,t] = 1`.
- `q[b,t]`: gold quality class index, valid only when `y[b,t] = 1`.

Change loss is weighted binary cross entropy over valid candidate slots:

```text
L_change = mean_m BCEWithLogits(s[b,t], y[b,t], pos_weight)
```

The root loss is cross entropy over the root vocabulary, but only at gold chord
change positions:

```text
root_mask[b,t] = m[b,t] and y[b,t] = 1 and r[b,t] is valid
L_root = mean_root_mask CE(root_logits[b,t], r[b,t])
```

The quality loss follows the same rule:

```text
quality_mask[b,t] = m[b,t] and y[b,t] = 1 and q[b,t] is valid
L_quality = mean_quality_mask CE(quality_logits[b,t], q[b,t])
```

The sparsity penalty discourages excessive chord changes:

```text
L_sparsity = mean_m p[b,t]
```

The temporal smoothness penalty discourages adjacent `1,1` change bursts:

```text
transition_mask[b,t] = m[b,t] and m[b,t-1]
L_smoothness = mean_transition_mask p[b,t] * p[b,t-1]
```

The total loss is:

```text
L_total =
  L_change
  + root_weight * L_root
  + quality_weight * L_quality
  + alpha * L_sparsity
  + beta * L_smoothness
```

Current starting coefficients:

```text
pos_weight = 1.5
root_weight = 1.0
quality_weight = 1.0
alpha = 0.1
beta = 0.3
```

Root and quality are intentionally not trained on hold positions. Hold slots
only affect `L_change`, `L_sparsity`, and `L_smoothness`; this prevents the
model from learning arbitrary root/quality classes where no chord event exists.

## Transposition Augmentation

Use `transpose_canonical_leadsheets.py` to create key-augmented canonical
copies. The script does not change the source files. It mechanically transposes:

- `context.key`
- chord roots and slash basses in `harmony_stream`
- melody pitches in `melody_stream`

It preserves bar, beat, duration, tags, meter, and raw stream shape.

Create all 12 transpositions for the OpenBook canonical snapshot:

```bash
python src/realbook_ingestion/scripts/transpose_canonical_leadsheets.py \
  --input data/openbook/canonical \
  --all-keys \
  --output-dir data/openbook/canonical_transposed_12key
```

Transpose one file to a target key:

```bash
python src/realbook_ingestion/scripts/transpose_canonical_leadsheets.py \
  --input data/openbook/canonical/autumn_leaves.canonical.json \
  --target-key "C minor" \
  --output-dir /tmp/autumn_leaves_transposed
```

Each generated file records transposition metadata in `context.transposition`.
For training splits, split by original tune identity before or during
augmentation so transposed copies of one tune do not leak across train/test
boundaries.

## Script

Build only the manifest and parsed event audit:

```bash
python src/realbook_ingestion/scripts/prepare_qwen_chord_tokenizer.py \
  --canonical-dir data/openbook/canonical \
  --output-dir test_logs/qwen25_15b_chord_tokenizer_manifest
```

Build an extended tokenizer from a downloaded Qwen tokenizer:

```bash
python src/realbook_ingestion/scripts/prepare_qwen_chord_tokenizer.py \
  --canonical-dir data/openbook/canonical \
  --tokenizer-path /root/autodl-tmp/models/Qwen2.5-1.5B-Base \
  --output-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Base-chord-tokenizer
```

Outputs:

- `tokenizer/`: Qwen tokenizer with appended chord special tokens
- `chord_vocab_manifest.json`: token list, root features, parser stats
- `parsed_harmony_events.jsonl`: one parsed row per canonical harmony event

Training with this tokenizer must resize the model embeddings:

```python
model.resize_token_embeddings(len(tokenizer))
```

For the No-LoRA 1.5B route, the final artifact should be a complete fine-tuned
model directory that includes this exact tokenizer.
