# Harmony Candidate Coverage Audit

## Summary

This report compares rule-based candidate mask slots against canonical `harmony_stream` chord onsets. It audits whether the model has a legal prediction slot at each gold chord-change position.

| Item | Value |
| --- | --- |
| Canonical source | `/Users/4paradigm/Documents/repos/jazzarranger/data/openbook/canonical` |
| Key filter | `ALL` |
| Candidate policy | `elastic_melody_rest_onset_plus_meter_backbone` |
| Tunes | `155` |
| Harmony events | `9892` |
| Covered events | `9758` |
| Missed events | `134` |
| Coverage | `98.65%` |

## Candidate Policy

Backbone beats by meter:

| Meter | Backbone beats |
| --- | --- |
| 4/4 | [1.0, 3.0, 4.0] |
| 2/2 | [1.0, 3.0, 4.0] |
| 3/4 | [1.0] |
| 5/4 | [1.0, 3.0, 4.0] |
| 6/8 | [1.0] |

Candidate type counts:

| Candidate type | Count |
| --- | --- |
| N | 20929 |
| C | 10501 |
| R | 1672 |

## Coverage By Meter

| Meter | Harmony events | Covered | Missed | Coverage |
| --- | --- | --- | --- | --- |
| 2/2 | 3534 | 3469 | 65 | 98.16% |
| 3/4 | 458 | 435 | 23 | 94.98% |
| 4/4 | 5820 | 5781 | 39 | 99.33% |
| 5/4 | 48 | 48 | 0 | 100.00% |
| 6/8 | 32 | 25 | 7 | 78.12% |

## Worst Tunes

| Title | File | Meter | Harmony events | Missed | Coverage |
| --- | --- | --- | --- | --- | --- |
| Greensleeves | greensleeves.canonical.json | 6/8 | 32 | 7 | 78.12% |
| Flamingo | flamingo.canonical.json | 4/4 | 57 | 10 | 82.46% |
| Don't Get Around Much Anymore | dont_get_around_much_anymore.canonical.json | 2/2 | 145 | 25 | 82.76% |
| Always | always.canonical.json | 3/4 | 51 | 7 | 86.27% |
| What A Diff'rence A Day Made | what_a_diffrence_a_day_made.canonical.json | 2/2 | 24 | 3 | 87.50% |
| Take the 'A' Train | take_the_a_train.canonical.json | 4/4 | 30 | 3 | 90.00% |
| Tenderly | tenderly.canonical.json | 3/4 | 40 | 4 | 90.00% |
| My One And Only Love | my_one_and_only_love.canonical.json | 2/2 | 173 | 12 | 93.06% |
| Guilty | guilty.canonical.json | 4/4 | 81 | 5 | 93.83% |
| Quiet Nights Of Quiet Stars | quiet_nights_of_quiet_stars.canonical.json | 2/2 | 37 | 2 | 94.59% |
| Body And Soul | body_and_soul.canonical.json | 4/4 | 58 | 3 | 94.83% |
| I Can't Give You Anything But Love | i_cant_give_you_anything_but_love.canonical.json | 4/4 | 44 | 2 | 95.45% |
| Alice In Wonderland | alice_in_wonderland.canonical.json | 3/4 | 68 | 3 | 95.59% |
| A Cup Of Coffee, A Sandwich And You | a_cup_of_coffee_a_sandwich_and_you.canonical.json | 2/2 | 91 | 4 | 95.60% |
| Waltz For Debby | waltz_for_debby.canonical.json | 3/4 | 92 | 4 | 95.65% |

## Missed Onset Distribution

| Beat | Missed count |
| --- | --- |
| 2 | 91 |
| 4.5 | 16 |
| 3 | 15 |
| 2.5 | 7 |
| 3.5 | 3 |
| 3.25 | 1 |
| 1.5 | 1 |

## Missed Examples

| Title | Bar | Beat | Chord | Nearest candidate |
| --- | --- | --- | --- | --- |
| A Cup Of Coffee, A Sandwich And You | 20 | 2 | G7 | {'step': 71, 'bar': 20, 'beat': 1, 'type': 'C', 'pitch': 'D5', 'note_duration': 0, 'distance_beats': -1.0} |
| A Cup Of Coffee, A Sandwich And You | 37 | 2 | F7 | {'step': 134, 'bar': 37, 'beat': 1, 'type': 'C', 'pitch': 'Bb4', 'note_duration': 0, 'distance_beats': -1.0} |
| A Cup Of Coffee, A Sandwich And You | 45 | 2 | F7 | {'step': 164, 'bar': 45, 'beat': 1, 'type': 'N', 'pitch': 'Bb4', 'note_duration': 3, 'distance_beats': -1.0} |
| A Cup Of Coffee, A Sandwich And You | 61 | 2 | F7 | {'step': 223, 'bar': 61, 'beat': 1, 'type': 'C', 'pitch': 'Bb4', 'note_duration': 0, 'distance_beats': -1.0} |
| A Night In Tunisia | 7 | 4.5 | Dm6 | {'step': 48, 'bar': 7, 'beat': 4, 'type': 'R', 'pitch': 'R', 'note_duration': 1, 'distance_beats': -0.5} |
| Alice In Wonderland | 15 | 2.5 | A7 | {'step': 33, 'bar': 15, 'beat': 2, 'type': 'N', 'pitch': 'G6', 'note_duration': 2, 'distance_beats': -0.5} |
| Alice In Wonderland | 16 | 2.5 | G7 | {'step': 35, 'bar': 16, 'beat': 3, 'type': 'N', 'pitch': 'F7', 'note_duration': 1, 'distance_beats': 0.5} |
| Alice In Wonderland | 47 | 3 | Ab7 | {'step': 108, 'bar': 47, 'beat': 2, 'type': 'N', 'pitch': 'E6', 'note_duration': 2, 'distance_beats': -1.0} |
| Always | 20 | 2 | F7/E | {'step': 62, 'bar': 20, 'beat': 2.5, 'type': 'N', 'pitch': 'Bb4', 'note_duration': 1, 'distance_beats': 0.5} |
| Always | 20 | 3 | F7/Eb | {'step': 62, 'bar': 20, 'beat': 2.5, 'type': 'N', 'pitch': 'Bb4', 'note_duration': 1, 'distance_beats': -0.5} |
| Always | 23 | 2 | Gm | {'step': 73, 'bar': 23, 'beat': 2.5, 'type': 'N', 'pitch': 'G4', 'note_duration': 1, 'distance_beats': 0.5} |
| Always | 42 | 3 | F7 | {'step': 131, 'bar': 43, 'beat': 1, 'type': 'C', 'pitch': 'A4', 'note_duration': 0, 'distance_beats': 1.0} |
| Always | 52 | 2 | F7/E | {'step': 162, 'bar': 52, 'beat': 1, 'type': 'N', 'pitch': 'F3', 'note_duration': 4, 'distance_beats': -1.0} |
| Always | 52 | 3 | F7/Eb | {'step': 163, 'bar': 53, 'beat': 1, 'type': 'C', 'pitch': 'F3', 'note_duration': 0, 'distance_beats': 1.0} |
| Always | 55 | 2 | Gm | {'step': 166, 'bar': 55, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': -1.0} |
| Angel Eyes | 4 | 2 | Abmaj7 | {'step': 25, 'bar': 4, 'beat': 1, 'type': 'C', 'pitch': 'G4', 'note_duration': 0, 'distance_beats': -1.0} |
| Angel Eyes | 12 | 2 | Abmaj7 | {'step': 80, 'bar': 12, 'beat': 1.75, 'type': 'N', 'pitch': 'Gb4', 'note_duration': 1.5, 'distance_beats': -0.25} |
| As Time Goes By | 32 | 2 | Db7 | {'step': 165, 'bar': 32, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': -1.0} |
| Bluesette | 120 | 3 | E7 | {'step': 369, 'bar': 121, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': 1.0} |
| Bluesette | 126 | 3 | D9 | {'step': 375, 'bar': 127, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': 1.0} |
| Bluesette | 127 | 2 | Am7 | {'step': 375, 'bar': 127, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': -1.0} |
| Bluesette | 127 | 3 | Abmaj7 | {'step': 376, 'bar': 128, 'beat': 1, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': 1.0} |
| Body And Soul | 19 | 2 | Bm7 | {'step': 114, 'bar': 19, 'beat': 1, 'type': 'C', 'pitch': 'A3', 'note_duration': 0, 'distance_beats': -1.0} |
| Body And Soul | 24 | 2 | B9 | {'step': 147, 'bar': 24, 'beat': 1, 'type': 'N', 'pitch': 'A4', 'note_duration': 1, 'distance_beats': -1.0} |
| Body And Soul | 25 | 2 | Db | {'step': 150, 'bar': 25, 'beat': 1, 'type': 'C', 'pitch': 'Db4', 'note_duration': 0, 'distance_beats': -1.0} |
| Chega De Saudade | 62 | 4.5 | A7/G | {'step': 283, 'bar': 62, 'beat': 4, 'type': 'N', 'pitch': 'B4', 'note_duration': 1, 'distance_beats': -0.5} |
| Chega De Saudade | 66 | 3.5 | A7 | {'step': 296, 'bar': 66, 'beat': 3, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': -0.5} |
| Could It Be You | 20 | 2 | Gbm7 | {'step': 84, 'bar': 20, 'beat': 1, 'type': 'C', 'pitch': 'F5', 'note_duration': 0, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 2 | 2 | C | {'step': 7, 'bar': 2, 'beat': 1, 'type': 'N', 'pitch': 'E4', 'note_duration': 4, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 2 | 4.5 | C/E | {'step': 9, 'bar': 2, 'beat': 4, 'type': 'C', 'pitch': 'E4', 'note_duration': 0, 'distance_beats': -0.5} |
| Don't Get Around Much Anymore | 4 | 2 | A | {'step': 16, 'bar': 4, 'beat': 1, 'type': 'N', 'pitch': 'E4', 'note_duration': 4, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 4 | 4.5 | A7/C# | {'step': 18, 'bar': 4, 'beat': 4, 'type': 'C', 'pitch': 'E4', 'note_duration': 0, 'distance_beats': -0.5} |
| Don't Get Around Much Anymore | 8 | 2 | C/E | {'step': 39, 'bar': 8, 'beat': 1, 'type': 'C', 'pitch': 'C4', 'note_duration': 0, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 10 | 2 | C | {'step': 50, 'bar': 10, 'beat': 1, 'type': 'C', 'pitch': 'E4', 'note_duration': 0, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 12 | 2 | A | {'step': 61, 'bar': 12, 'beat': 1, 'type': 'C', 'pitch': 'E4', 'note_duration': 0, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 16 | 2 | C/E | {'step': 83, 'bar': 16, 'beat': 1, 'type': 'C', 'pitch': 'C4', 'note_duration': 0, 'distance_beats': -1.0} |
| Don't Get Around Much Anymore | 16 | 4.5 | C | {'step': 85, 'bar': 16, 'beat': 4, 'type': 'C', 'pitch': 'R', 'note_duration': 0, 'distance_beats': -0.5} |
| Don't Get Around Much Anymore | 18 | 4.5 | Bb9 | {'step': 93, 'bar': 18, 'beat': 4, 'type': 'C', 'pitch': 'C5', 'note_duration': 0, 'distance_beats': -0.5} |
| Don't Get Around Much Anymore | 35 | 2 | C | {'step': 180, 'bar': 35, 'beat': 1.5, 'type': 'N', 'pitch': 'E5', 'note_duration': 1, 'distance_beats': -0.5} |
| Don't Get Around Much Anymore | 35 | 4.5 | C/E | {'step': 184, 'bar': 35, 'beat': 4, 'type': 'N', 'pitch': 'F4', 'note_duration': 1, 'distance_beats': -0.5} |

## Covered Examples

| Title | Bar | Beat | Chord | Candidate slot |
| --- | --- | --- | --- | --- |
| 500 Miles High | 1 | 1 | Em | {'step': 1, 'type': 'N', 'pitch': 'B5', 'note_duration': 4.666667} |
| 500 Miles High | 3 | 1 | Gm | {'step': 11, 'type': 'N', 'pitch': 'E5', 'note_duration': 5} |
| 500 Miles High | 5 | 1 | Bbmaj7 | {'step': 20, 'type': 'N', 'pitch': 'G4', 'note_duration': 4} |
| 500 Miles High | 7 | 1 | Bm7b5 | {'step': 27, 'type': 'C', 'pitch': 'E4', 'note_duration': 0} |
| 500 Miles High | 8 | 1 | E7#9 | {'step': 31, 'type': 'C', 'pitch': 'G4', 'note_duration': 0} |
| 500 Miles High | 9 | 1 | Am11 | {'step': 36, 'type': 'C', 'pitch': 'D5', 'note_duration': 0} |
| 500 Miles High | 11 | 1 | F#m7b5 | {'step': 44, 'type': 'C', 'pitch': 'D5', 'note_duration': 0} |
| 500 Miles High | 13 | 1 | Fm7 | {'step': 53, 'type': 'C', 'pitch': 'C5', 'note_duration': 0} |
| 500 Miles High | 15 | 1 | Cm11 | {'step': 63, 'type': 'C', 'pitch': 'C4', 'note_duration': 0} |
| 500 Miles High | 17 | 1 | B7b3b5b9b11b13 | {'step': 69, 'type': 'C', 'pitch': 'R', 'note_duration': 0} |
| 500 Miles High | 21 | 1 | A | {'step': 87, 'type': 'C', 'pitch': 'E4', 'note_duration': 0} |
| 500 Miles High | 25 | 1 | Em | {'step': 105, 'type': 'C', 'pitch': 'G3', 'note_duration': 0} |
| 500 Miles High | 27 | 1 | Gm | {'step': 113, 'type': 'C', 'pitch': 'D4', 'note_duration': 0} |
| 500 Miles High | 29 | 1 | Bbmaj7 | {'step': 120, 'type': 'C', 'pitch': 'D4', 'note_duration': 0} |
| 500 Miles High | 31 | 1 | Bm7b5 | {'step': 129, 'type': 'R', 'pitch': 'R', 'note_duration': 0.5} |
| 500 Miles High | 32 | 1 | E7#9 | {'step': 136, 'type': 'C', 'pitch': 'C3', 'note_duration': 0} |
| 500 Miles High | 33 | 1 | Am11 | {'step': 139, 'type': 'C', 'pitch': 'C3', 'note_duration': 0} |
| 500 Miles High | 35 | 1 | F#m7b5 | {'step': 145, 'type': 'C', 'pitch': 'R', 'note_duration': 0} |
| 500 Miles High | 37 | 1 | Fm7 | {'step': 156, 'type': 'C', 'pitch': 'C2', 'note_duration': 0} |
| 500 Miles High | 39 | 1 | Cm11 | {'step': 162, 'type': 'C', 'pitch': 'R', 'note_duration': 0} |

## Files

- `metadata.json`: audit configuration
- `summary.json`: aggregate coverage metrics
- `tune_summaries.jsonl`: one row per tune
- `harmony_event_comparison.jsonl`: one row per harmony event
- `missed_harmony_events.jsonl`: only uncovered harmony events
