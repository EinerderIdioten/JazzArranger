# Canonical Lead-Sheet Storage

This document defines the canonical storage format for Phase 1 Real Book-style jazz harmony experiments.

The canonical format is not the final format shown to a model. It is the source of truth for storage, validation, indexing, and scoring. Model-facing text should be generated from it by a temporal rendering layer.

```text
PDF / MusicXML / ABC / manual entry
  -> canonical lead-sheet storage
  -> validation and review
  -> temporal rendering for LLM prompts and RAG
  -> model output
  -> deterministic evaluation
```

## 1. Design goal

Phase 1 needs a compact representation for short jazz-ballad lead-sheet phrases:

- chord symbols
- single-line melody
- key, meter, tempo, style, and instrument metadata
- explicit bar and beat timing
- human review status
- benchmark input and reference output separation

The format is deliberately smaller than MusicXML and more explicit than ABC. It prioritizes deterministic software operations over direct visual rendering.

## 2. Canonical vs. rendered representations

There are two layers:

1. Canonical storage
2. Model-facing temporal rendering

Canonical storage keeps parallel streams:

```json
"harmony_stream": [
  [1, 1.0, "Cmaj7", 4.0]
],
"melody_stream": [
  [1, 1.0, "E4", 2.0, "structural"],
  [1, 3.0, "G4", 1.0, "structural"]
]
```

This is efficient for validation and database operations, but it separates simultaneous harmony and melody events into different arrays. Therefore, LLM prompts should normally use a generated temporal rendering such as event grids or bar-by-bar text. See [Model Temporal Rendering](model_temporal_rendering.md).

Rule:

```text
Canonical storage is maintained by humans and tools.
Model-facing views are generated from canonical storage.
```

## 3. Top-level lead-sheet structure

A canonical lead-sheet file uses this shape:

```json
{
  "schema": "lead_sheet.v0",
  "context": {
    "title": "Example Ballad Phrase",
    "key": "C major",
    "meter": "4/4",
    "tempo": 72,
    "style": "jazz_ballad",
    "instrument": "piano"
  },
  "stream_schema": {
    "harmony_stream": ["bar", "beat", "symbol", "duration_beats"],
    "melody_stream": ["bar", "beat", "pitch", "duration_beats", "tag"]
  },
  "harmony_stream": [
    [1, 1.0, "Dm7", 4.0],
    [2, 1.0, "G7", 4.0],
    [3, 1.0, "Cmaj7", 4.0]
  ],
  "melody_stream": [
    [1, 1.0, "F4", 2.0, "structural"],
    [1, 3.0, "E4", 1.0, "passing"],
    [1, 4.0, "D4", 1.0, "passing"],
    [2, 1.0, "B3", 2.0, "structural"],
    [2, 3.0, "D4", 1.0, "structural"],
    [2, 4.0, "F4", 1.0, "structural"],
    [3, 1.0, "E4", 4.0, "structural"]
  ],
  "review": {
    "status": "needs_review",
    "reviewer": null,
    "notes": []
  }
}
```

Canonical JSON files should be valid JSON without comments.

## 4. Context fields

Required fields:

- `title`: tune, phrase, or benchmark title
- `key`: notated or working key, such as `C major` or `G minor`
- `meter`: meter string, such as `4/4`
- `style`: Phase 1 normally uses `jazz_ballad`
- `instrument`: Phase 1 normally uses `piano`

Recommended fields:

- `tempo`: integer or approximate BPM
- `source_status`: source or rights status
- `phrase_scope`: bars or phrase identity
- `notes`: human-readable notes

## 5. Time model

Every event uses explicit bar-local coordinates:

```text
[bar, beat, value, duration_beats]
```

Rules:

- The first full notated bar is `bar = 1`.
- Pickup material before bar 1 uses `bar = 0`.
- `beat` is one-indexed inside the bar.
- `duration_beats` is measured in quarter-note beats.
- In 4/4, eighth notes use `0.5`, dotted quarter notes use `1.5`, and a full bar uses `4.0`.

Do not encode combined bar-position values such as `2.66`. Bar identity and beat position must remain separate.

## 6. Harmony stream

Each harmony event is:

```json
[bar, beat, symbol, duration_beats]
```

Example:

```json
[2, 1.0, "G7", 4.0]
```

For two chords in one bar:

```json
[
  [5, 1.0, "Bm7b5", 2.0],
  [5, 3.0, "E7alt", 2.0]
]
```

Chord symbols should preserve musical meaning. Normalize spelling conservatively. For example, only rewrite `C-7` to `Cm7` if the project explicitly adopts that convention.

## 7. Melody stream

Each melody event is:

```json
[bar, beat, pitch, duration_beats, tag]
```

Example:

```json
[1, 1.0, "F4", 2.0, "structural"]
```

Pitch uses spelled scientific pitch notation such as `C4`, `Db4`, `F#4`, `Bb4`. Use `R` for rests:

```json
[4, 4.0, "R", 1.0, "rest"]
```

Suggested tags:

- `pickup`
- `structural`
- `passing`
- `neighbor`
- `rest`
- `unknown`

Tags are not harmonic answers. Do not put chord-tone roles such as `3`, `b7`, `9`, or `13` into raw melody streams. These roles belong in a derived analysis layer.

## 8. Optional derived analysis

After raw transcription is reviewed, tools may generate analysis:

```json
{
  "analysis": {
    "melody_roles": [
      {
        "bar": 1,
        "beat": 1.0,
        "pitch": "F4",
        "source_chord": "Dm7",
        "role": "b3",
        "importance": "structural"
      }
    ],
    "harmonic_functions": [
      {
        "bar": 1,
        "beat": 1.0,
        "symbol": "Dm7",
        "function": "ii in C major"
      }
    ]
  }
}
```

Analysis is derived, not raw transcription. A benchmark runner should support hard mode with raw harmony and melody only, and assisted mode with derived melody roles and harmonic functions.

## 9. Benchmark wrapper

A benchmark case wraps canonical lead-sheet input and reference output:

```json
{
  "schema": "reharm_benchmark.v0",
  "case_id": "example_ballad_001",
  "level": 1,
  "task_type": "jazz_ballad_reharmonization",
  "input": {
    "context": {
      "key": "C major",
      "meter": "4/4",
      "tempo": 72,
      "style": "jazz_ballad",
      "instrument": "piano"
    },
    "harmony_stream": [
      [1, 1.0, "Cmaj7", 4.0],
      [2, 1.0, "Am7", 4.0],
      [3, 1.0, "Dm7", 4.0],
      [4, 1.0, "G7", 4.0]
    ],
    "melody_stream": [
      [1, 1.0, "E4", 2.0, "structural"],
      [1, 3.0, "G4", 1.0, "structural"],
      [2, 1.0, "B4", 2.0, "structural"],
      [3, 1.0, "F4", 2.0, "structural"],
      [4, 3.0, "F4", 2.0, "structural"]
    ]
  },
  "reference_output": {
    "reharmonized_harmony_stream": [
      [1, 1.0, "Cmaj9", 2.0],
      [1, 3.0, "E7b9", 2.0],
      [2, 1.0, "Am9", 2.0],
      [2, 3.0, "A7b13", 2.0],
      [3, 1.0, "Dm9", 2.0],
      [3, 3.0, "Db7#11", 2.0],
      [4, 1.0, "Cmaj9/G", 2.0],
      [4, 3.0, "G13sus", 2.0]
    ]
  }
}
```

The reference answer should remain separate from the raw lead-sheet input.

## 10. Validation checklist

A valid canonical file should satisfy:

- JSON syntax is valid.
- `schema` is present and recognized.
- Required context fields are present.
- Every harmony event has 4 fields.
- Every melody event has 5 fields.
- Bar numbers are integers.
- Beats and durations are positive numbers.
- Pitches match spelled scientific pitch notation or `R`.
- Chord symbols are non-empty strings.
- Melody events do not overlap in a monophonic phrase.
- Harmony events cover the intended phrase without unintended gaps.
- Raw melody streams do not contain derived chord-tone roles.

## 11. Relationship to MusicXML and ABC

This format does not replace MusicXML or ABC.

Use MusicXML when the goal is notation fidelity, rendering, and interchange with notation software.

Use ABC when the goal is compact human authoring or quick melody-plus-chord sketches.

Use canonical lead-sheet JSON when the goal is validation, indexing, RAG metadata, deterministic scoring, model input/output contracts, and separation between raw transcription and derived analysis.
