# Real Book Ingestion and Lead-Sheet Representation

This folder defines the Phase 1 representation strategy for Real Book-style jazz-ballad harmony experiments.

Phase 1 focuses on piano-only lead-sheet reharmonization:

```text
simple/source harmony + single-line melody
  -> richer jazz-ballad harmony
```

The project separates storage from model presentation:

```text
MusicXML / ABC / manual transcription
  -> canonical lead-sheet storage
  -> generated temporal rendering
  -> LLM prompt / RAG chunk / evaluator
```

## Scope

Confirmed Phase 1 scope:

- Instrument: piano only.
- Style: jazz ballad.
- Source material shape: Real Book-style lead sheets with chord symbols and single-line melody.
- Model task: given simple/source harmony plus melody, produce richer jazz harmony.
- Benchmark comparison: Base LLM vs Base LLM + RAG on the same sample set.

Out of scope for this documentation pass:

- PDF extraction implementation.
- Full arrangement extraction.
- Drum, groove, or bass transcription.
- Fine-tuning or model training.
- Score rendering.

## Documents

- [Canonical Lead-Sheet Storage](canonical_storage.md): source-of-truth JSON structure for storage, validation, indexing, and deterministic scoring.
- [Model Temporal Rendering](model_temporal_rendering.md): conversion from canonical storage into event grids, compact text, prompt views, and RAG chunks for model consumption.
- [OpenBook Mako Conversion](openbook_mako_conversion.md): static converter route from OpenBook `.ly.mako` symbolic lead sheets into canonical lead-sheet JSON.

## Design rule

Canonical storage is optimized for software correctness. Model-facing rendering is optimized for temporal readability.

Do not force one format to solve both problems.

```text
harmony_stream + melody_stream = canonical source of truth

event_grid / compact_text / prompt_view = generated derived views
```

## Why this split exists

Parallel canonical streams with absolute bar-beat coordinates are good for validation and scoring, but they can separate simultaneous events in the text. For example, a chord event and a melody event that happen on the same beat may live in different arrays.

A model should usually see a rendered time-ordered view instead:

```text
Bar 1:
- Beat 1.0: Cmaj7, melody E4 for 2 beats, structural
- Beat 3.0: Cmaj7 continues, melody G4 for 1 beat, structural
```

This keeps retrieval and validation deterministic while making the musical timeline easier for smaller LLMs to follow.

## Licensing note

Only ingest and store source material that the project has the right to use. If copyrighted lead sheets are used for private research, keep raw PDF files outside public version control and store only project-approved derived benchmark data.
