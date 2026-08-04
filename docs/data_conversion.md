# Data Conversion Notes

This project stores raw downloaded datasets under `data/raw/` and source archives
under `data/downloads/`. These paths are ignored by Git.

Processed training records are written as JSONL under `data/processed/`.

Each record uses the stage-one schema:

```json
{
  "id": "dataset:source_id",
  "dataset": "EMOPIA+",
  "dataset_weight": 1.0,
  "split": "train",
  "abc_melody": "X:...\nM:4/4\nL:1/16\nK:C\n...",
  "grid_resolution": "1/16",
  "time_signature": "4/4",
  "key": "C:maj",
  "total_grid": 64,
  "chords": [
    {"start": 0, "end": 16, "root": "C", "quality": "maj", "raw": "0_M"}
  ],
  "harmony": "<HARMONY>\n@0-16 C:maj\n</HARMONY>"
}
```

## Dataset Order

Initial conversion order follows the current supervision weights:

1. EMOPIA+ (`1.0`)
2. HLSD (`1.0`)
3. POP909 (`0.7`)
4. OpenBook (`pending converter audit`)

## EMOPIA+

EMOPIA+ lead-sheet files are REMI pickle files. The converter reads:

- `Bar`
- `Beat`
- `Chord`
- `Note_Pitch`
- `Note_Duration`

The observed beat grid is `0..15` per 4/4 bar, with duration `120` matching
one grid unit. The first converter therefore emits `L:1/16` ABC and
`grid_resolution="1/16"` to avoid losing odd 16th-note positions.

`None_None` chord files are skipped for the first supervised target because
stage one does not yet define a no-chord root/quality class.

## HLSD

The local HLSD zip currently contains the repository sample data, not the 4.9GB
full source archive. The converter therefore processes the available sample
`datasets/event_list.json` entries first.

HLSD event JSON files expose:

- `metadata`
- `tracks.melody`
- `tracks.chord`
- `num_measures`

Event times are beat offsets. The converter maps one beat to four grid units,
so processed records also use `L:1/16` and `grid_resolution="1/16"`.

HLSD melody pitch values are numeric symbolic pitches rather than MIDI note
numbers in the sample event files. The first converter maps them into the middle
ABC register with `MIDI = 60 + pitch`, preserving relative melodic contour for
stage-one data plumbing.

## POP909

POP909 is converted after EMOPIA+ and HLSD because its dataset weight is `0.7`.
The converter reads each song directory under `data/raw/pop909/POP909`:

- `<song_id>.mid`
- `beat_midi.txt`
- `chord_midi.txt`
- `key_audio.txt`

The main MIDI file contains a `MELODY` track. The converter uses a small
standard-library MIDI parser instead of depending on `miditoolkit` or
`pretty_midi`, then maps note ticks through the MIDI tempo map into seconds and
onto `beat_midi.txt`.

POP909 beat rows are treated as quarter-note beat positions. The second column
of `beat_midi.txt` is used to infer the measure length, which is usually `2/4`
and sometimes `3/4` in the local files. Processed output still uses `L:1/16`
with four grid units per quarter note.

`chord_midi.txt` labels are canonicalized into the stage-one root/quality
vocabulary. Slash/inversion annotations and extensions are downgraded, for
example:

- `Bb:maj/5` -> `A#:maj`
- `C:7/b7` -> `C:dom7`
- `F#:maj7/5` -> `F#:maj7`
- `C:sus4(b7)` -> `C:sus4`

`N` labels are not represented as training labels in stage one. POP909 records
are therefore split into continuous canonical-chord segments; leading/trailing
or internal no-chord areas are excluded from the supervised rows.

## OpenBook

OpenBook is last because the current plan marks it as pending converter audit.
The local version is a LilyPond/Mako lead-sheet repository rather than a clean
event dataset. The first converter reads `src/openbook/*.ly.mako`, prefers
`ChordsFake`/`VoiceFake` pairs, and falls back to `ChordsReal`/`VoiceReal`.

The parser handles common LilyPond note/chord spelling and downgrades common
extensions into the stage-one vocabulary. Files with tuplets or grace-note
syntax are skipped for now because the shared `1/16` grid cannot represent them
exactly without a defined quantization policy.

Converted OpenBook rows are emitted with `dataset_weight=0.0` and
`conversion_status="audit_pending"` so they can be inspected without being
accidentally mixed into the main supervised training loss.

## Commands

Run converters from the repository root:

```bash
python3 -m src.data.convert_emopia_plus
python3 -m src.data.convert_hlsd
python3 -m src.data.convert_pop909
python3 -m src.data.convert_openbook
python3 -m src.data.build_conversion_summary
```
