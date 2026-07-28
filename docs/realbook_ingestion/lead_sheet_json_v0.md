# Lead-Sheet JSON v0

Lead-Sheet JSON v0 is a compact canonical format for Phase 1 jazz-ballad benchmark data. It records the objective lead-sheet content needed for reharmonization: metadata, chord symbols, melody pitches, durations, and bar-local timing.

The format is intentionally smaller than MusicXML and less ambiguous than MIDI. It is designed to be easy to review by humans and easy to normalize into richer internal models later.

## Design principles

1. Store objective transcription first; derive music-theory analysis later.
2. Use explicit bar and beat coordinates instead of packed floating-point positions.
3. Keep streams compact enough for manual correction.
4. Keep canonical files valid JSON with no comments.
5. Preserve pitch spelling such as `F#4` vs `Gb4` because spelling affects harmonic interpretation.
6. Do not put answer-like melody roles into the raw input stream unless a benchmark mode explicitly requires assisted input.

## Top-level structure

```json
{
  "schema": "lead_sheet.v0",
  "context": {
    "title": "Example Ballad Fragment",
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
    [3, 1.0, "Cmaj7", 4.0],
    [4, 1.0, "Fmaj7", 4.0]
  ],
  "melody_stream": [
    [0, 2.5, "A4", 0.5, "pickup"],
    [0, 3.0, "B4", 0.5, "pickup"],
    [0, 3.5, "C5", 0.5, "pickup"],
    [1, 1.0, "F4", 4.0, "structural"],
    [2, 2.5, "E4", 0.5, "pickup"],
    [2, 3.0, "F4", 0.5, "pickup"],
    [2, 3.5, "G4", 0.5, "pickup"],
    [3, 1.0, "E4", 4.0, "structural"]
  ],
  "review": {
    "status": "needs_review",
    "reviewer": null,
    "notes": []
  }
}
```

## Time model

Use bar-local coordinates:

```text
[bar, beat, value, duration_beats]
```

Rules:

- The first full notated bar is `bar = 1`.
- Pickup material before bar 1 uses `bar = 0`.
- `beat` is one-indexed inside the bar: `1.0`, `2.0`, `3.0`, `4.0` in 4/4.
- `duration_beats` is measured in quarter-note beats.
- In 4/4, a full-bar whole note has `duration_beats = 4.0`.
- Eighth notes have `duration_beats = 0.5`.
- Dotted quarter notes have `duration_beats = 1.5`.

Avoid packed positions like `2.66` because they mix bar identity, beat location, and fractional notation in one ambiguous number.

## Harmony stream

Each harmony event is:

```json
[bar, beat, symbol, duration_beats]
```

Example:

```json
[2, 1.0, "D7", 4.0]
```

Meaning:

- Bar 2.
- Beat 1.
- Chord symbol `D7`.
- Lasts 4 beats.

For two chords in one bar:

```json
[
  [5, 1.0, "Bm7b5", 2.0],
  [5, 3.0, "E7alt", 2.0]
]
```

## Melody stream

Each melody event is:

```json
[bar, beat, pitch, duration_beats, tag]
```

Example:

```json
[1, 1.0, "C5", 4.0, "structural"]
```

`pitch` should use spelled scientific pitch notation:

```text
C4, Db4, D4, Eb4, E4, F4, F#4, G4, Ab4, A4, Bb4, B4
```

Allowed accidentals in v0:

```text
bb, b, #, ##
```

Use `R` for rests:

```json
[4, 4.0, "R", 1.0, "rest"]
```

Suggested tags:

- `pickup`: pickup or approach material.
- `structural`: important sustained or target melody note.
- `passing`: passing tone.
- `neighbor`: neighbor tone.
- `rest`: rest event.
- `unknown`: not yet reviewed.

Tags are transcription aids, not full harmonic analysis. For example, use `structural` rather than `3` or `13` in raw input. Chord-tone roles should be generated in an analysis layer.

## Optional analysis layer

After raw transcription is validated, an analyzer can add derived fields:

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

The benchmark runner should be able to render prompts with or without this analysis:

- Hard mode: give only raw harmony and melody.
- Assisted mode: include derived melody roles and harmonic functions.

Phase 1 should start with hard mode to avoid leaking the answer.

## Benchmark case extension

Raw lead-sheet JSON can be wrapped into a benchmark case:

```json
{
  "schema": "reharm_benchmark.v0",
  "case_id": "example_ballad_001",
  "input": {
    "context": {},
    "harmony_stream": [],
    "melody_stream": []
  },
  "reference_output": {
    "reharmonized_harmony_stream": [
      [1, 1.0, "Dm9", 4.0],
      [2, 1.0, "G13", 4.0],
      [3, 1.0, "Cmaj9", 4.0],
      [4, 1.0, "Fmaj9", 4.0]
    ]
  }
}
```

The lead-sheet transcription and the benchmark answer should remain separate so that the same input can be evaluated against different reference reharmonization styles later.

## Validation checklist

A valid v0 file should satisfy:

- `schema` equals `lead_sheet.v0`.
- `context.title`, `context.key`, and `context.meter` are present.
- Every harmony event has 4 fields.
- Every melody event has 5 fields.
- Bar numbers are integers.
- Beats and durations are positive numbers.
- Pitches match spelled pitch notation or `R`.
- Chord symbols are non-empty strings.
- No melody events overlap unless polyphonic melody is explicitly enabled in a later schema.
- Chords cover the intended phrase without unintended gaps.
