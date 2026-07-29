# Real Book to Lead-Sheet JSON

This folder defines the Phase 1 ingestion plan for turning Real Book-style lead sheets into compact, structured JSON that can be used for jazz-ballad reharmonization benchmarks.

Phase 1 focuses on two goals:

1. Design a structured data format for Real Book-style simple harmony and melody in jazz ballads.
2. Design a batch processing workflow that can combine automated extraction with human correction.

## Scope

Confirmed Phase 1 scope:

- Instrument: piano only.
- Style: jazz ballad.
- Source material shape: Real Book-style lead sheets with chord symbols and single-line melody.
- Model task: given simple harmony plus melody, produce richer jazz harmony.
- Benchmark comparison: Base LLM vs Base LLM + RAG on the same sample set.

Out of scope for Phase 1:

- Full arrangement extraction.
- Drum/groove transcription.
- Bass line transcription.
- Fully automatic PDF recognition without human review.
- Fine-tuning or training a model.

## Documents

- [Lead-sheet JSON v0](lead_sheet_json_v0.md): compact canonical data structure for harmony and melody.
- [PDF to JSON workflow](pdf_to_json_workflow.md): batch processing route from PDF or MusicXML into validated JSON.
- [Compact lead-sheet notation report](compact_lead_sheet_notation_report.md): rationale, examples, and comparison with MusicXML and ABC.

## Recommended data path

```text
Real Book-style PDF
  -> page images or MusicXML where available
  -> OMR / manual compact transcription
  -> normalized lead-sheet JSON
  -> validation and human review
  -> benchmark case generation
```

The canonical dataset should be pure JSON without comments. Human-facing notes, review state, and provenance should live in metadata fields rather than JSON comments.

## Licensing note

Only ingest and store source material that the project has the right to use. If copyrighted lead sheets are used for private research, keep raw PDF files outside public version control and store only project-approved derived benchmark data.
