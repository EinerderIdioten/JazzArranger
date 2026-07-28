# PDF to JSON Workflow

This document defines a practical Phase 1 workflow for converting Real Book-style lead sheets into validated Lead-Sheet JSON v0. The goal is not full automation at any cost. The goal is reliable benchmark data with a clear place for human correction.

## Target output

The batch pipeline should produce one reviewed JSON file per tune or phrase:

```text
data/lead_sheets/<tune_slug>.json
```

Each file should follow [Lead-Sheet JSON v0](lead_sheet_json_v0.md).

## Recommended pipeline

```text
1. Source registration
2. PDF page extraction
3. OMR or manual transcription
4. Normalization to Lead-Sheet JSON v0
5. Automated validation
6. Human review and correction
7. Benchmark-case extraction
```

## 1. Source registration

Maintain a local manifest outside public distribution if source PDFs are copyrighted:

```json
{
  "source_id": "local_realbook_volume_01_page_012",
  "title": "Example Tune",
  "source_type": "pdf",
  "local_path": "sources/private/realbook/page_012.pdf",
  "rights_status": "private_research",
  "ingestion_status": "not_started"
}
```

The manifest is useful for tracking review state, but raw copyrighted PDFs should not be committed unless the project has redistribution rights.

## 2. PDF page extraction

Convert each relevant PDF page to an image for OMR or manual review:

```text
sources/private/pdf_pages/<source_id>.png
```

Recommended preprocessing:

- Crop page margins.
- Deskew scanned pages.
- Increase contrast if notation is faint.
- Keep one tune per image when possible.

## 3. OMR or manual transcription

Use the fastest reliable path for each source.

### Route A: MusicXML available

```text
MusicXML
  -> parse chord symbols and melody
  -> normalize to Lead-Sheet JSON v0
  -> human review
```

This is the best route when a clean MusicXML file exists.

### Route B: OMR-assisted

```text
PDF image
  -> OMR tool
  -> MusicXML
  -> correction in notation software
  -> normalize to JSON
```

Possible tools include Audiveris or commercial OMR software. OMR output should always be reviewed because lead-sheet chord symbols, handwritten fonts, repeats, endings, and pickup bars are common failure points.

### Route C: Manual compact transcription

For short jazz-ballad benchmark phrases, manual transcription can be faster and more accurate than OMR.

A human can enter compact streams directly:

```json
{
  "harmony_stream": [
    [1, 1.0, "Dm7", 4.0],
    [2, 1.0, "G7", 4.0]
  ],
  "melody_stream": [
    [0, 2.5, "A4", 0.5, "pickup"],
    [0, 3.0, "B4", 0.5, "pickup"],
    [1, 1.0, "F4", 4.0, "structural"]
  ]
}
```

For Phase 1, Route C is acceptable and often preferable for the initial 10-30 benchmark phrases.

## 4. Normalization

The normalizer should make the output consistent regardless of source route.

Required normalization:

- Convert chord symbols to a consistent spelling style.
- Convert note names to spelled scientific pitch notation.
- Convert all durations to quarter-note beats.
- Use bar `0` for pickup material.
- Expand obvious repeat notation only when needed for benchmark phrases.
- Keep raw transcription separate from derived analysis.

Chord symbol spelling should be normalized conservatively. Do not rewrite musical meaning during ingestion. For example, preserve `C-7` as `Cm7` only if the project explicitly adopts that chord-symbol style.

## 5. Automated validation

The validator should check structural correctness before human review:

- JSON syntax is valid.
- Schema is `lead_sheet.v0`.
- Required context fields exist.
- All stream rows have the expected number of fields.
- Pitch strings are valid.
- Chord symbols are non-empty.
- Durations are positive.
- Beats fit inside the meter except for explicitly allowed pickup conventions.
- Melody events do not overlap in the monophonic melody stream.
- Harmony events cover each benchmark bar as expected.

Validation should produce actionable messages:

```text
bar 2 beat 3.5: melody duration extends beyond bar boundary
bar 4: missing harmony coverage from beat 3.0 to 5.0
bar 5 beat 1.0: invalid pitch spelling H4
```

## 6. Human review

Human review is part of the pipeline, not a failure of automation.

Recommended review fields:

```json
{
  "review": {
    "status": "needs_review",
    "reviewer": null,
    "notes": []
  }
}
```

Allowed statuses:

- `needs_review`
- `reviewed`
- `approved`
- `rejected`

Human review checklist:

1. Title, key, and meter match the source.
2. Pickup bar is represented correctly.
3. Chord symbols are placed on the correct beats.
4. Melody pitches use correct enharmonic spelling.
5. Durations match the notation.
6. Repeats, first/second endings, and codas are either resolved or deliberately excluded.
7. No analytical role labels are mixed into the raw stream unless intentional.
8. Phrase boundaries for benchmark extraction are clear.

## 7. Benchmark-case extraction

After approval, extract short phrases for Phase 1 comparison:

```text
approved lead-sheet JSON
  -> select 4-bar or 8-bar phrase
  -> simplify or preserve source harmony
  -> attach reference reharmonization
  -> run Base LLM and Base LLM + RAG
```

The benchmark case should separate input from reference output:

```json
{
  "schema": "reharm_benchmark.v0",
  "case_id": "example_001_a_section_bars_1_4",
  "input": {
    "context": {},
    "harmony_stream": [],
    "melody_stream": []
  },
  "reference_output": {
    "reharmonized_harmony_stream": []
  }
}
```

## Batch directory layout

A practical local layout:

```text
sources/
  private/
    pdf/
    images/
    musicxml/
data/
  lead_sheets/
    raw/
    reviewed/
    approved/
  benchmark_cases/
    phase1/
reports/
  ingestion_validation/
  human_review/
```

The `sources/private` directory should usually remain outside public version control. Approved JSON files can be versioned if they do not contain restricted source material beyond what the project is allowed to store.

## Phase 1 recommendation

Start with manual compact transcription for a small curated set:

```text
10-30 phrases
4 or 8 bars each
jazz ballad only
piano reharmonization target
```

This keeps the first benchmark focused on model comparison instead of spending the entire phase on imperfect PDF recognition. After the benchmark runner and validator are stable, add OMR-assisted ingestion to increase scale.
