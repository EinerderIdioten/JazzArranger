# Compact Lead-Sheet JSON Notation Report

This report describes the project's compact lead-sheet notation for Real Book-style jazz harmony experiments and compares it with MusicXML and ABC notation.

The goal is not to replace existing music notation ecosystems. The goal is to define a small, machine-scoreable, LLM-friendly intermediate format for Phase 1 experiments:

```text
Real Book-style melody + chord symbols
  -> compact lead-sheet JSON
  -> benchmark prompt / RAG document / evaluator input
```

## 1. Why define another notation layer?

MusicXML and ABC already have ecosystems. They are useful for notation exchange, rendering, and tool interoperability. However, the Phase 1 benchmark has a narrower goal:

- represent a short lead-sheet phrase
- preserve chord symbols
- preserve melody pitch spelling
- align melody events with bar-local beats
- keep the format short enough to fit cleanly inside LLM prompts
- support deterministic validation and scoring
- keep raw transcription separate from derived harmonic analysis

The project therefore uses compact JSON streams as a canonical benchmark format, while still allowing import/export adapters from MusicXML, MIDI, ABC, or manual transcription.

## 2. Core format

The canonical format is JSON. Each file stores objective lead-sheet content first:

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
    [3, 1.0, "Cmaj7", 4.0],
    [4, 1.0, "Cmaj7", 4.0]
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

### 2.1 Harmony event

Each harmony event has four fields:

```json
[bar, beat, symbol, duration_beats]
```

Example:

```json
[2, 1.0, "G7", 4.0]
```

Meaning:

- bar 2
- beat 1.0
- chord symbol `G7`
- lasts 4 quarter-note beats

### 2.2 Melody event

Each melody event has five fields:

```json
[bar, beat, pitch, duration_beats, tag]
```

Example:

```json
[1, 1.0, "F4", 2.0, "structural"]
```

Meaning:

- bar 1
- beat 1.0
- pitch `F4`
- lasts 2 quarter-note beats
- transcription tag is `structural`

The `tag` field is not harmonic analysis. It is only a transcription aid. For example:

- `structural`
- `passing`
- `neighbor`
- `pickup`
- `rest`
- `unknown`

Do not put chord-tone answers such as `3`, `b7`, or `13` into raw melody events. Those should be derived later by an analyzer.

## 3. Timing model

Use explicit bar-local timing:

```text
[bar, beat, value, duration_beats]
```

Rules:

- The first full bar is `bar = 1`.
- Pickup material before bar 1 uses `bar = 0`.
- Beat is one-indexed inside the bar: `1.0`, `2.0`, `3.0`, `4.0` in 4/4.
- Durations are measured in quarter-note beats.
- Avoid packed positions such as `2.66`, because they mix bar number and beat position.

Example with pickup:

```json
{
  "harmony_stream": [
    [1, 1.0, "Dm7", 4.0],
    [2, 1.0, "G7", 4.0]
  ],
  "melody_stream": [
    [0, 3.0, "A4", 0.5, "pickup"],
    [0, 3.5, "B4", 0.5, "pickup"],
    [1, 1.0, "C5", 2.0, "structural"]
  ]
}
```

## 4. Benchmark wrapper

For model evaluation, the lead-sheet input can be wrapped in a benchmark case:

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

This wrapper keeps input and reference answer separate. The same input can later be evaluated against different reference styles.

## 5. Comparison with MusicXML

MusicXML is the most complete option for exchanging notated scores with notation programs.

### 5.1 MusicXML strengths

- Large ecosystem: MuseScore, Finale, Dorico, Sibelius import/export paths.
- Good for score rendering and full notation fidelity.
- Can represent meter, key signatures, articulations, layout, repeats, lyrics, slurs, voices, and multi-staff piano notation.
- Suitable for long-term archival interchange.

### 5.2 MusicXML weaknesses for this benchmark

- Verbose for LLM prompts.
- Harder to manually review in raw text form.
- Chord symbols and melody alignment can be buried inside a large tree structure.
- Different exporters produce different MusicXML shapes.
- Full notation fidelity is more than Phase 1 needs.

### 5.3 Approximate MusicXML shape

The same idea in MusicXML becomes much longer:

```xml
<measure number="1">
  <harmony>
    <root><root-step>D</root-step></root>
    <kind>minor-seventh</kind>
  </harmony>
  <note>
    <pitch>
      <step>F</step>
      <octave>4</octave>
    </pitch>
    <duration>2</duration>
    <type>half</type>
  </note>
</measure>
```

For rendering, this is better than compact JSON. For model prompts and benchmark fixtures, it is usually too verbose.

## 6. Comparison with ABC notation

ABC is a compact text notation with a real ecosystem, especially for folk, traditional, and melody-centric music.

### 6.1 ABC strengths

- Very compact.
- Human-readable after learning the syntax.
- Has existing parsers and renderers.
- Good for single-line melody and chord-symbol sketches.
- Easy to write by hand.

### 6.2 ABC weaknesses for this benchmark

- Timing and bar alignment require parsing notation syntax rather than reading direct fields.
- Enharmonic spelling and octave notation are compact but can be error-prone for LLMs.
- Less direct support for benchmark metadata, review state, expected outputs, and evaluation fields.
- Good for authoring, but less explicit as a canonical scoring object.

### 6.3 Approximate ABC shape

A compact ABC phrase may look like:

```abc
X:1
T:Example Ballad Phrase
M:4/4
L:1/4
K:C
"Dm7" F2 E D | "G7" B,2 D F | "Cmaj7" E4 |
```

This is concise and useful for humans. But for automated evaluation, the system still has to parse positions, durations, pitches, and chord spans into structured events.

## 7. Why compact JSON is useful for LLM evaluation

The compact JSON stream is designed for benchmark operations:

| Requirement | Compact JSON | MusicXML | ABC |
| --- | --- | --- | --- |
| Short prompt footprint | Good | Poor | Excellent |
| Human editability | Good | Poor | Good |
| Existing notation ecosystem | Weak | Excellent | Good |
| Direct JSON validation | Excellent | Good with XML schema | Requires parser |
| Explicit bar-beat coordinates | Excellent | Available but nested | Requires parsing |
| Chord symbol + melody alignment | Excellent | Good but verbose | Good after parsing |
| Benchmark metadata | Excellent | Awkward | Awkward |
| Score rendering | Poor | Excellent | Good |
| LLM structured output compatibility | Excellent | Moderate | Moderate |

The recommended approach is therefore not exclusive:

```text
MusicXML / ABC / manual transcription
  -> compact lead-sheet JSON
  -> validation
  -> benchmark prompt and evaluator
```

## 8. Proposed PDF-to-JSON role

For Real Book-style PDF ingestion, compact JSON should be the normalized target after OCR/OMR and human review.

```text
PDF page image
  -> OMR or manual transcription
  -> MusicXML or compact text if useful
  -> normalized compact lead-sheet JSON
  -> validation
  -> human review
  -> benchmark cases
```

A practical converter does not need perfect PDF understanding on day one. For Phase 1, a human-in-the-loop workflow is more reliable:

1. Extract the page image.
2. Use OMR or manual entry to capture chords and melody.
3. Convert to compact JSON streams.
4. Run structural validation.
5. Human reviewer checks chord placement, pitch spelling, durations, and phrase boundaries.
6. Generate benchmark cases from approved phrases.

## 9. Validation checklist

A valid compact lead-sheet JSON phrase should satisfy:

- JSON syntax is valid.
- `schema` is present.
- `context.key`, `context.meter`, and `context.style` are present.
- Every harmony event has 4 fields.
- Every melody event has 5 fields.
- Bar numbers are integers.
- Beats and durations are positive numbers.
- Pitches use scientific pitch notation or `R` for rest.
- Chord symbols are non-empty strings.
- Melody events do not overlap in a monophonic phrase.
- Chord events cover the phrase without unintended gaps.
- Raw melody streams do not contain derived chord-tone roles.

## 10. Recommendation

Use compact JSON as the canonical internal benchmark format, but do not treat it as a replacement for MusicXML or ABC.

Recommended policy:

- Use MusicXML when the priority is notation fidelity and ecosystem interoperability.
- Use ABC or compact text when the priority is fast human entry.
- Use compact lead-sheet JSON when the priority is benchmark evaluation, LLM prompting, RAG indexing, validation, and scoring.

This lets the project benefit from existing notation ecosystems while still keeping Phase 1 evaluation data small, explicit, and machine-scoreable.
