# Harmony-Rhythm Training Strategy

This document defines the Phase 1 strategy for training and evaluating a model
to predict chord-change timing from canonical lead-sheet data.

## Pipeline Step

This strategy is a model-facing rendering step between canonical storage and
fine-tuning/evaluation:

```text
canonical lead-sheet storage
  -> harmony-rhythm candidate rendering
  -> zero-shot JSON prompt or token-level classifier rendering
  -> model output / gathered mask-token logits
  -> deterministic scoring against harmony_stream
```

Canonical `harmony_stream` remains the source of truth for labels. It must not
be included in the model input for this task.

## Task

Given melody timing checkpoints, predict whether each checkpoint should start a
new chord.

The model does not predict chord symbols in this phase. It only predicts
harmony-rhythm labels:

```text
1 = change to a new chord at this checkpoint
0 = continue the current harmony
```

## Candidate Time Points

The model can only predict chord changes at time points that appear in the
input. Therefore the renderer uses an elastic mixed candidate grid rather than
a fixed 0.5-beat grid or a pure melody-onset grid.

A candidate is generated when any of these conditions is true:

- Condition A: a melody note onset or rest onset occurs.
- Condition B: a bar backbone point occurs.
- Condition C: a late-bar preparation point occurs for common-time tunes.

Default backbone policy:

```text
4/4, 2/2: beats 1, 3, and 4
3/4:      beat 1
other:    beat 1, plus beats 3 and 4 when the meter has at least 4 quarter-note beats
```

Condition C adds beat 4 for `4/4` and `2/2`. In the current OpenBook-derived
canonical snapshot this implementation covers `9758 / 9892` harmony changes,
or about `98.6%`, while adding a modest number of candidates.

## Candidate Row Format

Each model input row is:

```json
[bar, beat, type, pitch, duration_beats, "M"]
```

`type` values:

```text
N = new pitched melody onset
R = rest onset
C = continuation backbone checkpoint
```

For `C` rows, `pitch` is the currently sounding melody pitch when known, or
`R` if no melody event is active. `duration_beats` is `0` for continuation
checkpoints because no new melody event starts there.

Example input:

```json
[
  [1, 1, "N", "C5", 1, "M"],
  [1, 3, "C", "C5", 0, "M"],
  [1, 4, "C", "C5", 0, "M"],
  [2, 1, "N", "D5", 2, "M"]
]
```

Expected output:

```json
[1, 0, 1, 1]
```

For zero-shot generation, the output must be a JSON array of `0`/`1` integers
with the same length as the input row count.

For fine-tuning, do not train the model to autoregressively generate the JSON
array. Use the fixed-length token-classifier route described in
[Harmony-Rhythm Token Classifier](harmony_rhythm_token_classifier.md).

## Prompt Template

System prompt:

```text
你是一个爵士乐编曲器。输入是一个二维数组，每行代表一个旋律时间检查点 [小节, 拍数, 类型, 音高, 时值, "M"]。请直接输出一个相同长度的一维 JSON 数组，用 1 或 0 替换 "M"。1 代表该检查点换新和弦，0 代表不换。不要输出任何多余的解释。
```

User prompt:

```text
类型说明：N=新音符开始，R=休止开始，C=骨架延续检查点。
输入长度: {n}
输入:
{rows_json}
仅输出包含 {n} 个数字的 JSON 数组。
```

## Label Extraction

Labels are generated from `harmony_stream`:

```json
[bar, beat, symbol, duration_beats]
```

For each candidate row:

- label `1` if a harmony event starts at the same `(bar, beat)`.
- label `0` otherwise.

Harmony events that do not land on any candidate are counted as uncovered by
the candidate policy. They are not shown to the model and should be reported as
candidate coverage loss.

## Evaluation

Treat this as binary sequence labeling, not text generation.

Required metrics:

- format pass rate: valid JSON array, exact expected length, only `0`/`1`
- precision, recall, and F1 for label `1`
- strict candidate-position match
- optional tolerant match within `0.5` beat
- candidate coverage: covered harmony changes / total harmony changes

Invalid outputs are scored as all-zero predictions in aggregate metrics so that
format failures are penalized.

The JSON-generation baseline is retained only as a pre-finetuning diagnostic.
The production training path gathers `0`/`1` logits at fixed label-slot mask
positions, so output length is enforced by code instead of by prompting.

## Data Split

Training and evaluation splits must be tune-level splits, not window-level
splits. Different windows from the same tune must not appear in both train and
test sets.
