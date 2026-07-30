# OpenBook Mako Conversion

This document describes the first-pass converter from OpenBook `.ly.mako`
lead-sheet sources into JazzArranger canonical lead-sheet JSON.

## Why This Route Exists

OpenBook sources are already symbolic music files. For tunes covered by
OpenBook, this is a better ingestion route than optical music recognition:

```text
OpenBook .ly.mako
  -> extracted metadata, chord section, and voice section
  -> canonical lead-sheet JSON
  -> review and benchmark preparation
```

The converter is intentionally conservative. It extracts objective lead-sheet
content and marks generated files as `needs_review`.

## Source Format

OpenBook `.ly.mako` files combine two layers:

- Mako template guards such as `% if part=='Vars':`.
- LilyPond music fragments inside sections such as `ChordsReal` and `VoiceReal`.

The `Vars` section contains metadata:

- `attributes['title']`
- `attributes['composer']`
- `attributes['style']`
- `attributes['piece']`
- `attributes['structure']`
- `attributes.set_default_version_name(...)`

Musical material is split by version:

- `ChordsReal`, `ChordsFake`, `ChordsAebersold`, etc.
- `VoiceReal`, `VoiceFake`, `VoiceAebersold`, etc.
- Optional lyric sections are currently ignored by canonical storage.

## Canonical Output

The output follows [Canonical Lead-Sheet Storage](canonical_storage.md):

```json
{
  "schema": "lead_sheet.v0",
  "context": {},
  "stream_schema": {
    "harmony_stream": ["bar", "beat", "symbol", "duration_beats"],
    "melody_stream": ["bar", "beat", "pitch", "duration_beats", "tag"]
  },
  "harmony_stream": [],
  "melody_stream": [],
  "review": {
    "status": "needs_review",
    "reviewer": null,
    "notes": []
  }
}
```

## Script Location

```text
src/realbook_ingestion/scripts/openbook_mako_to_canonical.py
```

The script has no third-party Python dependencies.

## Repository Data Snapshot

The checked-in OpenBook ingestion snapshot lives under:

```text
data/openbook/
  raw/        original OpenBook .ly.mako files
  canonical/ generated lead_sheet.v0 JSON files
```

Regenerate the canonical snapshot whenever `raw/` changes:

```bash
python3 -B src/realbook_ingestion/scripts/openbook_mako_to_canonical.py \
  data/openbook/raw \
  -o data/openbook/canonical
```

## Single-File Conversion

```bash
python3 src/realbook_ingestion/scripts/openbook_mako_to_canonical.py \
  ~/Desktop/openbook-master/src/openbook/autumn_leaves.ly.mako \
  --version Fake \
  -o /private/tmp/autumn_leaves.canonical.json
```

If `--version` is omitted, the script uses the OpenBook default version when it
has both chords and melody. If the default version has an empty stream and
another version is complete, the script falls back and records that decision in
`review.notes`.

## Batch Conversion

```bash
python3 src/realbook_ingestion/scripts/openbook_mako_to_canonical.py \
  ~/Desktop/openbook-master/src/openbook \
  -o /private/tmp/openbook_canonical_test
```

Batch mode requires `--output` to be a directory. Each input file is written as:

```text
<tune_name>.canonical.json
```

## Supported LilyPond Subset

The converter currently handles the OpenBook lead-sheet subset needed for first
benchmark ingestion:

- `\partial`
- inherited note and chord durations
- dotted durations
- multiplied durations such as `c1*2:maj7`
- `\repeat volta`
- `\alternative`
- `\repeat unfold`
- `\tuplet` and `\times`
- rests and skips
- basic LilyPond relative pitch notation with OpenBook's outer `\relative c'`
- chord roots, chord qualities, alterations, and slash basses

Example mappings:

```text
a:m7       -> Am7
fis:m7.5-  -> F#m7b5
b:7.9-     -> B7b9
bes:maj7   -> Bbmaj7
e:m/d      -> Em/D
```

## Known Limitations

This is not a complete LilyPond interpreter.

The converter ignores score-rendering commands that are not part of canonical
storage, such as layout breaks, coda/segno glyphs, markups, and dynamics. These
are recorded in `review.notes` when encountered.

Generated melody tags are mechanical:

- `pickup` for bar `0`
- `rest` for rests
- `unknown` for ordinary melody notes

Human review should still mark structural, passing, and neighbor tones before a
benchmark case is finalized.

## Validation Performed

The initial implementation was checked against the local OpenBook
`src/openbook` folder:

- 155 `.ly.mako` files converted without CLI failure.
- No converted file had an empty `harmony_stream` or `melody_stream`.
- No stream event had an out-of-range beat coordinate for its meter.

This validation only checks structural conversion. It does not replace musical
review.
