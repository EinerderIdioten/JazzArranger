"""Convert POP909 MIDI melody + chord annotations into ABC melody + spans."""

from __future__ import annotations

import argparse
import bisect
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.common import (
    ChordSpan,
    NoteEvent,
    canonicalize_root_quality,
    chord_spans_to_harmony,
    counter_to_sorted_dict,
    ensure_dirs,
    melody_notes_to_abc,
    normalize_root_name,
    validate_chord_spans,
    validate_harmony_text,
    write_json,
    write_jsonl,
)


DATASET_NAME = "POP909"
DATASET_WEIGHT = 0.7
GRIDS_PER_BEAT = 4


@dataclass(frozen=True)
class MidiNote:
    start_tick: int
    end_tick: int
    pitch: int


@dataclass(frozen=True)
class MidiTrack:
    name: str
    notes: list[MidiNote]


@dataclass(frozen=True)
class MidiFile:
    ticks_per_beat: int
    time_signature: tuple[int, int] | None
    tempo_events: list[tuple[int, int]]
    tracks: list[MidiTrack]


@dataclass(frozen=True)
class BeatGrid:
    times: list[float]
    beats_per_bar: int


@dataclass(frozen=True)
class TimedChord:
    start_time: float
    end_time: float
    start_grid: int
    end_grid: int
    raw: str
    root: str | None
    quality: str | None


def read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def parse_midi(path: Path) -> MidiFile:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("missing_midi_header")
    header_len = struct.unpack(">I", data[4:8])[0]
    _, track_count, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("smpte_midi_time_division")
    offset = 8 + header_len

    tempo_events: list[tuple[int, int]] = []
    time_signature: tuple[int, int] | None = None
    tracks: list[MidiTrack] = []

    for _ in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError("missing_track_header")
        length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        track_data = data[offset + 8 : offset + 8 + length]
        offset += 8 + length

        pos = 0
        tick = 0
        running_status: int | None = None
        track_name = ""
        active: dict[tuple[int, int], list[int]] = defaultdict(list)
        notes: list[MidiNote] = []

        while pos < len(track_data):
            delta, pos = read_varlen(track_data, pos)
            tick += delta
            status = track_data[pos]
            if status < 0x80:
                if running_status is None:
                    raise ValueError("running_status_without_status")
                status = running_status
            else:
                pos += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                meta_type = track_data[pos]
                pos += 1
                length, pos = read_varlen(track_data, pos)
                payload = track_data[pos : pos + length]
                pos += length
                if meta_type == 0x03:
                    track_name = payload.decode("latin1", errors="ignore")
                elif meta_type == 0x51 and len(payload) == 3:
                    tempo_events.append((tick, int.from_bytes(payload, "big")))
                elif meta_type == 0x58 and len(payload) >= 2 and time_signature is None:
                    time_signature = (payload[0], 2 ** payload[1])
                continue

            if status in (0xF0, 0xF7):
                length, pos = read_varlen(track_data, pos)
                pos += length
                continue

            event_type = status & 0xF0
            channel = status & 0x0F
            if event_type in (0xC0, 0xD0):
                pos += 1
                continue

            pitch = track_data[pos]
            velocity = track_data[pos + 1]
            pos += 2
            note_key = (channel, pitch)
            if event_type == 0x90 and velocity > 0:
                active[note_key].append(tick)
            elif event_type in (0x80, 0x90):
                starts = active.get(note_key)
                if starts:
                    start_tick = starts.pop(0)
                    if tick > start_tick:
                        notes.append(MidiNote(start_tick=start_tick, end_tick=tick, pitch=pitch))

        tracks.append(MidiTrack(name=track_name, notes=notes))

    if not tempo_events:
        tempo_events = [(0, 500000)]
    tempo_events = sorted(tempo_events)
    if tempo_events[0][0] != 0:
        tempo_events.insert(0, (0, 500000))
    return MidiFile(
        ticks_per_beat=division,
        time_signature=time_signature,
        tempo_events=tempo_events,
        tracks=tracks,
    )


def seconds_at_tick(tick: int, midi: MidiFile) -> float:
    seconds = 0.0
    prev_tick = 0
    tempo = midi.tempo_events[0][1]
    for event_tick, event_tempo in midi.tempo_events[1:]:
        if tick < event_tick:
            break
        seconds += (event_tick - prev_tick) * tempo / 1_000_000 / midi.ticks_per_beat
        prev_tick = event_tick
        tempo = event_tempo
    seconds += (tick - prev_tick) * tempo / 1_000_000 / midi.ticks_per_beat
    return seconds


def read_beat_grid(path: Path) -> BeatGrid:
    times: list[float] = []
    downbeat_candidates: list[int] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        if float(parts[1]) == 1.0:
            downbeat_candidates.append(idx)
    if len(times) < 2:
        raise ValueError("not_enough_beat_rows")

    beats_per_bar = 4
    if len(downbeat_candidates) >= 2:
        diffs = [
            downbeat_candidates[idx + 1] - downbeat_candidates[idx]
            for idx in range(len(downbeat_candidates) - 1)
            if downbeat_candidates[idx + 1] > downbeat_candidates[idx]
        ]
        if diffs:
            beats_per_bar = Counter(diffs).most_common(1)[0][0]
    return BeatGrid(times=times, beats_per_bar=beats_per_bar)


def time_to_grid(time_value: float, beat_grid: BeatGrid) -> int:
    times = beat_grid.times
    if time_value <= times[0]:
        beat_step = times[1] - times[0]
        beat_pos = (time_value - times[0]) / beat_step
    elif time_value >= times[-1]:
        beat_step = times[-1] - times[-2]
        beat_pos = len(times) - 1 + (time_value - times[-1]) / beat_step
    else:
        idx = bisect.bisect_right(times, time_value) - 1
        beat_step = times[idx + 1] - times[idx]
        beat_pos = idx + (time_value - times[idx]) / beat_step
    return int(round(beat_pos * GRIDS_PER_BEAT))


def read_pop909_key(path: Path, segment_start_time: float) -> str:
    selected = "C:maj"
    if not path.exists():
        return selected
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        if start <= segment_start_time < end:
            selected = parts[2]
            break
    if ":" not in selected:
        return "C:maj"
    root_raw, mode_raw = selected.split(":", 1)
    root = normalize_root_name(root_raw)
    mode = "min" if mode_raw.startswith("min") else "maj"
    return f"{root or 'C'}:{mode}"


def canonicalize_pop909_chord(raw: str) -> tuple[str, str] | None:
    raw = raw.strip()
    if raw == "N" or ":" not in raw:
        return None
    root_raw, quality_raw = raw.split(":", 1)
    root = normalize_root_name(root_raw)
    if root is None:
        return None

    quality_base = quality_raw.split("/", 1)[0]
    quality_base = re.sub(r"\([^)]*\)", "", quality_base)
    downgrade = {
        "maj6": "maj",
        "min6": "min",
        "minmaj7": "min",
    }
    quality_base = downgrade.get(quality_base, quality_base)
    return canonicalize_root_quality(root, quality_base)


def read_chords(path: Path, beat_grid: BeatGrid) -> list[TimedChord]:
    chords: list[TimedChord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        start_time = float(parts[0])
        end_time = float(parts[1])
        raw = parts[2]
        canonical = canonicalize_pop909_chord(raw)
        root, quality = canonical if canonical else (None, None)
        chords.append(
            TimedChord(
                start_time=start_time,
                end_time=end_time,
                start_grid=time_to_grid(start_time, beat_grid),
                end_grid=time_to_grid(end_time, beat_grid),
                raw=raw,
                root=root,
                quality=quality,
            )
        )
    return chords


def melody_track(midi: MidiFile) -> MidiTrack | None:
    for track in midi.tracks:
        if track.name.strip().upper() == "MELODY" and track.notes:
            return track
    note_tracks = [track for track in midi.tracks if track.notes]
    return note_tracks[0] if note_tracks else None


def midi_notes_to_grid_notes(midi: MidiFile, notes: list[MidiNote], beat_grid: BeatGrid) -> list[NoteEvent]:
    grid_notes: list[NoteEvent] = []
    for note in notes:
        start = time_to_grid(seconds_at_tick(note.start_tick, midi), beat_grid)
        end = time_to_grid(seconds_at_tick(note.end_tick, midi), beat_grid)
        if end <= start:
            end = start + 1
        grid_notes.append(NoteEvent(start=start, end=end, pitch=note.pitch))
    return sorted(grid_notes, key=lambda item: (item.start, item.end, item.pitch))


def split_for_song(song_id: str) -> str:
    try:
        value = int(song_id)
    except ValueError:
        value = sum(ord(char) for char in song_id)
    bucket = value % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def chord_segments(chords: list[TimedChord]) -> list[list[TimedChord]]:
    segments: list[list[TimedChord]] = []
    current: list[TimedChord] = []
    for chord in chords:
        if chord.root is None or chord.quality is None or chord.end_grid <= chord.start_grid:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(chord)
    if current:
        segments.append(current)
    return segments


def rebase_spans(segment: list[TimedChord]) -> tuple[int, int, list[ChordSpan]]:
    start_grid = segment[0].start_grid
    end_grid = segment[-1].end_grid
    spans: list[ChordSpan] = []
    for chord in segment:
        assert chord.root is not None and chord.quality is not None
        start = chord.start_grid - start_grid
        end = chord.end_grid - start_grid
        if end <= start:
            continue
        if spans and spans[-1].root == chord.root and spans[-1].quality == chord.quality and spans[-1].end == start:
            prev = spans[-1]
            spans[-1] = ChordSpan(prev.start, end, prev.root, prev.quality, prev.raw)
        else:
            spans.append(ChordSpan(start=start, end=end, root=chord.root, quality=chord.quality, raw=chord.raw))
    return start_grid, end_grid, spans


def clip_notes_to_segment(notes: list[NoteEvent], start_grid: int, end_grid: int) -> list[NoteEvent]:
    clipped: list[NoteEvent] = []
    for note in notes:
        start = max(note.start, start_grid)
        end = min(note.end, end_grid)
        if end > start:
            clipped.append(NoteEvent(start=start - start_grid, end=end - start_grid, pitch=note.pitch))
    return clipped


def convert_song(song_dir: Path) -> tuple[list[dict], list[dict]]:
    song_id = song_dir.name
    failures: list[dict] = []
    try:
        beat_grid = read_beat_grid(song_dir / "beat_midi.txt")
        midi = parse_midi(song_dir / f"{song_id}.mid")
        track = melody_track(midi)
        if track is None:
            return [], [{"id": song_id, "errors": ["missing_melody_track"]}]
        notes = midi_notes_to_grid_notes(midi, track.notes, beat_grid)
        timed_chords = read_chords(song_dir / "chord_midi.txt", beat_grid)
    except Exception as exc:
        return [], [{"id": song_id, "errors": [f"parse_error:{type(exc).__name__}:{exc}"]}]

    rows: list[dict] = []
    bar_grid = beat_grid.beats_per_bar * GRIDS_PER_BEAT
    time_signature = f"{beat_grid.beats_per_bar}/4"
    split = split_for_song(song_id)
    segments = chord_segments(timed_chords)
    if not segments:
        return [], [{"id": song_id, "errors": ["missing_canonical_chord_segments"]}]

    for idx, segment in enumerate(segments):
        start_grid, end_grid, spans = rebase_spans(segment)
        total_grid = end_grid - start_grid
        errors = validate_chord_spans(spans, total_grid)
        segment_notes = clip_notes_to_segment(notes, start_grid, end_grid)
        if not segment_notes:
            errors.append("missing_notes")
        if total_grid <= 0:
            errors.append("empty_total_grid")
        if errors:
            failures.append(
                {
                    "id": song_id,
                    "segment": idx,
                    "source_grid": [start_grid, end_grid],
                    "errors": errors,
                }
            )
            continue

        first_time = segment[0].start_time
        key = read_pop909_key(song_dir / "key_audio.txt", first_time)
        title = f"pop909_{song_id}_seg{idx}"
        abc_melody, abc_stats = melody_notes_to_abc(
            segment_notes,
            total_grid=total_grid,
            title=title,
            key=key,
            time_signature=time_signature,
            unit_length="1/16",
            bar_grid=bar_grid,
        )
        harmony = chord_spans_to_harmony(spans)
        harmony_errors = validate_harmony_text(harmony)
        if harmony_errors:
            failures.append(
                {
                    "id": song_id,
                    "segment": idx,
                    "source_grid": [start_grid, end_grid],
                    "errors": harmony_errors,
                }
            )
            continue

        rows.append(
            {
                "id": f"pop909:{song_id}:seg{idx}",
                "dataset": DATASET_NAME,
                "dataset_weight": DATASET_WEIGHT,
                "split": split,
                "source_path": str(song_dir),
                "abc_melody": abc_melody,
                "grid_resolution": "1/16",
                "time_signature": time_signature,
                "key": key,
                "total_grid": total_grid,
                "chords": [span.to_json() for span in spans],
                "harmony": harmony,
                "metadata": {
                    "song_id": song_id,
                    "segment_index": idx,
                    "source_grid_start": start_grid,
                    "source_grid_end": end_grid,
                    "source_chord_count": len(segment),
                    "beats_per_bar": beat_grid.beats_per_bar,
                    "midi_time_signature": midi.time_signature,
                },
                "conversion": abc_stats,
            }
        )
    return rows, failures


def convert(raw_root: Path, output_dir: Path, log_dir: Path) -> dict:
    ensure_dirs(output_dir, log_dir)
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []
    quality_counter: Counter = Counter()
    root_counter: Counter = Counter()
    split_counter: Counter = Counter()
    conversion_counter: Counter = Counter()

    song_dirs = sorted([path for path in raw_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    for song_dir in song_dirs:
        rows, song_failures = convert_song(song_dir)
        failures.extend(song_failures)
        for failure in song_failures:
            for error in failure["errors"]:
                conversion_counter[error.split(":", 1)[0]] += 1
        for row in rows:
            rows_by_split[row["split"]].append(row)
            split_counter[row["split"]] += 1
            conversion_counter["converted"] += 1
            conversion_counter["dropped_same_start_notes"] += row["conversion"]["dropped_same_start_notes"]
            conversion_counter["truncated_overlapping_notes"] += row["conversion"]["truncated_overlapping_notes"]
            for chord in row["chords"]:
                quality_counter[chord["quality"]] += 1
                root_counter[chord["root"]] += 1

    for split in ("train", "val", "test"):
        rows = rows_by_split.get(split, [])
        if rows:
            write_jsonl(output_dir / f"pop909_{split}.jsonl", rows)

    report = {
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "total_files": len(song_dirs),
        "converted_records": conversion_counter["converted"],
        "failed_segments": len(failures),
        "split_counts": counter_to_sorted_dict(split_counter),
        "quality_distribution": counter_to_sorted_dict(quality_counter),
        "root_distribution": counter_to_sorted_dict(root_counter),
        "conversion_stats": counter_to_sorted_dict(conversion_counter),
        "failures": failures[:300],
    }
    write_json(log_dir / "pop909_conversion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/pop909/POP909"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    args = parser.parse_args()
    report = convert(args.raw_root, args.output_dir, args.log_dir)
    print(
        f"POP909 converted={report['converted_records']} failed_segments={report['failed_segments']} "
        f"splits={report['split_counts']}"
    )


if __name__ == "__main__":
    main()
