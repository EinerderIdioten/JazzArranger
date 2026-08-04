# OpenBook Lead-Sheet Data

This directory stores the OpenBook source files used for first-pass symbolic
lead-sheet ingestion and the generated JazzArranger canonical JSON files.

## Layout

```text
data/openbook/
  LICENSE.openbook
  raw/
    *.ly.mako
  canonical/
    *.canonical.json
```

## Raw Source

`raw/` contains OpenBook `.ly.mako` files copied from:

```text
openbook-master/src/openbook
```

These files are symbolic Mako plus LilyPond lead-sheet sources. They are not
PDFs and do not require OMR.

## Canonical JSON

`canonical/` contains files generated from `raw/` with:

```bash
python3 -B src/realbook_ingestion/scripts/openbook_mako_to_canonical.py \
  data/openbook/raw \
  -o data/openbook/canonical
```

Each generated file follows the project canonical storage shape:

```text
schema = lead_sheet.v0
harmony_stream = [bar, beat, symbol, duration_beats]
melody_stream = [bar, beat, pitch, duration_beats, tag]
```

Stream events are written as compact single-line arrays inside otherwise
pretty-printed JSON, so repeated musical events remain readable in diffs and
model prompts.

Generated files are marked `needs_review`. The converter preserves objective
lead-sheet events, but a human should still review melody tags, repeats, and
musical details before using a tune as a benchmark case.

## Validation Snapshot

The current data snapshot contains:

- 155 raw `.ly.mako` files.
- 155 generated `.canonical.json` files.
- No empty harmony or melody streams in the generated files.
- No out-of-range beat coordinates in the generated files.

## License

`LICENSE.openbook` is copied from the local OpenBook source tree. Keep license
and provenance information with these source-derived files.
