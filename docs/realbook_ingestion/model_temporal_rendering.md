# Model Temporal Rendering

This document defines how to convert canonical lead-sheet storage into model-facing sequential formats.

Canonical storage uses absolute bar-beat coordinates in parallel `harmony_stream` and `melody_stream` arrays. This is good for validation and indexing, but it can be awkward for LLMs because simultaneous musical events may appear far apart in the JSON text.

The solution is a generated temporal rendering layer:

```text
canonical harmony_stream + melody_stream
  -> event_grid
  -> compact_text / prompt_view / REMI-like tokens
  -> LLM or RAG
```

## 1. Design principle

Canonical storage remains the source of truth. Temporal renderings are derived views.

Rules:

- Do not hand-maintain derived views as canonical data.
- Generate derived views at runtime or during dataset build.
- Use canonical JSON for validation and scoring.
- Use rendered temporal views for model prompts, embeddings, and human debugging.

This avoids the main weakness of the raw parallel-stream format: the model does not have to mentally join simultaneous chord and melody events across separate arrays.

## 2. Why rendering is needed

Canonical storage example:

```json
{
  "harmony_stream": [
    [1, 1.0, "Cmaj7", 4.0]
  ],
  "melody_stream": [
    [1, 1.0, "E4", 2.0, "structural"],
    [1, 3.0, "G4", 1.0, "structural"],
    [1, 4.0, "A4", 1.0, "passing"]
  ]
}
```

For a program, this is easy to query. For a model, the simultaneity between `Cmaj7` and `E4` is less visually obvious.

A rendered model view should present the same information in time order:

```text
Bar 1:
- Beat 1.0: Cmaj7, melody E4 for 2 beats, structural
- Beat 3.0: Cmaj7 continues, melody G4 for 1 beat, structural
- Beat 4.0: Cmaj7 continues, melody A4 for 1 beat, passing
```

## 3. Event grid

The first derived form is `event_grid`. It groups all relevant active events at each time point.

```json
{
  "event_grid": [
    {
      "bar": 1,
      "beat": 1.0,
      "active_chord": "Cmaj7",
      "chord_started": true,
      "melody_events": [
        {
          "pitch": "E4",
          "duration_beats": 2.0,
          "tag": "structural"
        }
      ]
    },
    {
      "bar": 1,
      "beat": 3.0,
      "active_chord": "Cmaj7",
      "chord_started": false,
      "melody_events": [
        {
          "pitch": "G4",
          "duration_beats": 1.0,
          "tag": "structural"
        }
      ]
    }
  ]
}
```

`event_grid` may include one melody event in Phase 1 because the melody is monophonic. The array form leaves room for future polyphonic melody or ornament layers.

## 4. Conversion algorithm

Input:

- `harmony_stream`: `[bar, beat, symbol, duration_beats]`
- `melody_stream`: `[bar, beat, pitch, duration_beats, tag]`
- meter information from `context.meter`

Output:

- ordered event grid
- optional compact text
- optional prompt view
- optional token stream

Algorithm:

1. Validate canonical streams.
2. Convert every event start to a sortable coordinate `(bar, beat)`.
3. Build a sorted set of event-grid coordinates from every melody onset, every chord onset, and optional bar starts.
4. For each coordinate, find the active chord whose interval contains the coordinate.
5. Collect melody events that start at that coordinate.
6. Mark whether a new chord starts at that coordinate.
7. Sort by `(bar, beat)`.
8. Render the event grid into one or more model-facing views.

Active chord lookup:

```text
chord is active at coordinate T if:
  chord_start <= T < chord_start + duration_beats
```

The implementation should use meter-aware coordinate arithmetic rather than raw decimal concatenation.

## 5. Compact text rendering

Compact text is a concise, human-readable sequence for prompts and RAG chunks.

Example:

```text
Bar 1 | Cmaj7: E4/2@1 structural; G4/1@3 structural; A4/1@4 passing |
```

Recommended syntax:

```text
Bar <bar> | <chord>: <pitch>/<duration>@<beat> <tag>; ... |
```

For chord changes inside a bar:

```text
Bar 5 | Bm7b5: D4/1@1 structural; F4/1@2 structural | E7alt: G#4/1@3 structural; B4/1@4 tension |
```

This format is not canonical. It is allowed to be optimized for readability and prompt length.

## 6. Prompt view rendering

Prompt view is more verbose than compact text and is useful for weaker models.

```text
Context: C major, 4/4, slow jazz ballad, piano.

Bar 1:
- Beat 1.0: chord Cmaj7 starts. Melody E4 lasts 2 beats. E is structural.
- Beat 3.0: chord Cmaj7 continues. Melody G4 lasts 1 beat. G is structural.
- Beat 4.0: chord Cmaj7 continues. Melody A4 lasts 1 beat. A is passing.

Task: reharmonize the phrase while preserving the melody.
```

Use prompt view when the model struggles with compact notation.

## 7. REMI-like token rendering

A token stream can be useful for future symbolic-model training or strict sequence testing.

```text
[Bar_1] [Beat_1] [Chord_Cmaj7_Start] [Melody_E4] [Dur_2] [Tag_structural]
[Beat_3] [Chord_Cmaj7_Continue] [Melody_G4] [Dur_1] [Tag_structural]
[Beat_4] [Chord_Cmaj7_Continue] [Melody_A4] [Dur_1] [Tag_passing]
```

This is not the recommended first prompt format for Qwen3-8B. It is more useful later if the project tests sequence encodings or trainable symbolic models.

## 8. RAG chunk design

RAG should not embed raw canonical JSON alone. The recommended chunk stores canonical data plus rendered and analytical text.

```json
{
  "chunk_id": "example_ballad_001_bars_1_4",
  "metadata": {
    "title": "Example Ballad Phrase",
    "key": "C major",
    "style": "jazz_ballad",
    "bars": [1, 4],
    "concepts": ["ii-V-I", "secondary dominant", "major 9 color"]
  },
  "canonical": {
    "harmony_stream": [],
    "melody_stream": []
  },
  "compact_text": "Bar 1 | Cmaj7: E4/2@1 structural; G4/1@3 structural | ...",
  "analysis_text": "The phrase starts in tonic major color and moves toward a dominant cadence. Structural melody notes emphasize E, G, B, F."
}
```

Embedding priority:

1. `analysis_text`
2. `compact_text`
3. selected metadata
4. not raw JSON alone

The raw canonical object should still be stored for exact validation and reconstruction.

## 9. Model prompt contract

A benchmark runner should produce prompts using this structure:

```text
You are given a lead-sheet phrase rendered from canonical storage.
Preserve every melody note. Reharmonize for slow jazz-ballad piano.
Return strict JSON in the requested output schema.

Context:
<key, meter, tempo, style, instrument>

Lead-sheet phrase:
<compact_text or prompt_view>

Return:
{
  "reharmonized_harmony_stream": [
    [bar, beat, "ChordSymbol", duration_beats]
  ],
  "explanation": ["..."],
  "risk_notes": ["..."]
}
```

The model output should return canonical-style coordinates so that the evaluator can score it deterministically.

## 10. Evaluation round trip

The complete loop is:

```text
canonical input
  -> generated temporal prompt
  -> model output as canonical-style reharmonized_harmony_stream
  -> validator checks JSON shape and timing
  -> harmony evaluator checks melody compatibility and function
  -> human or automated rubric scores the answer
```

This preserves both advantages:

- absolute coordinates for machine scoring
- sequential temporal text for model comprehension

## 11. Practical recommendation for Phase 1

For Qwen3-8B and similar models, start with prompt view or compact text rather than raw streams.

Suggested progression:

1. language screening questions
2. prompt view with explicit beat-by-beat chord/melody alignment
3. compact text once the model proves it can follow the format
4. raw canonical JSON only for stronger models or tool-using agents

Canonical JSON should remain the database and evaluator format throughout all stages.
