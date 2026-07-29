# Realbook Language Screening Evaluation

- Benchmark: `realbook_benchmark/language_screening_questions.json`
- Model: `qwen3-8b`
- Serving stack: vLLM OpenAI-compatible API
- API used during run: `http://127.0.0.1:18000/v1`
- Run time: `2026-07-29T14:46:22+08:00` to `2026-07-29T14:46:39+08:00`
- Scoring rubric: `0 = wrong or irrelevant`, `1 = partially correct`, `2 = correct and clear`

## Summary

Total score: **17 / 20**

Result: **Pass**. The benchmark defines `14/20` as the pass threshold for trying harder reharmonization cases. The model understands most basic Real Book / jazz-ballad language, but it showed weakness on detailed extension spelling and avoid-note theory.

## Scores

| ID | Topic | Score | Assessment |
| --- | --- | ---: | --- |
| `lang_001` | chord symbol meaning | 2 | Correctly identifies Cmaj7 as C major with major seventh and gives C, E, G, B. |
| `lang_002` | dominant function | 2 | Correctly identifies G7 as dominant-function harmony resolving to C; guide tones are not named but the answer is clear. |
| `lang_003` | ii-V-I | 2 | Correctly gives Dm7-G7-Cmaj7 and describes common jazz motion to tonic. |
| `lang_004` | guide tones | 2 | Correctly names 3rd and 7th and explains their role in outlining changes. |
| `lang_005` | rootless voicing | 2 | Correctly says the root is omitted and upper structures/extensions are emphasized, with bass-line context. |
| `lang_006` | tensions | 1 | Correctly says 9, 11, and 13 add color/tension, but describes their intervals incorrectly as distances above the 7th. |
| `lang_007` | avoid notes | 1 | Correctly identifies a clash with the 3rd, but misidentifies the natural 11th/#11 pitch content. |
| `lang_008` | tritone substitution | 2 | Correctly gives Db7 for G7 and explains chromatic resolution to Cmaj7. |
| `lang_009` | melody compatibility | 1 | Correctly says both chords can support E, but does not state the key chord-tone facts: E is 3rd of Cmaj7 and 5th of Am7. |
| `lang_010` | ballad pacing | 2 | Correctly explains that too many fast complex changes can clutter a slow ballad and reduce melodic space. |

## Notable Issues

1. `lang_006`: The answer treats 9, 11, and 13 as intervals above the 7th. For jazz chord symbols, they should be understood in chord context as compound 2nd, 4th, and 6th above the root.
2. `lang_007`: The answer says the natural 11th in Cmaj7 is notated as `D`, and says the sharp 11th is `D#`. The expected pitches are `F` for natural 11 and `F#` for sharp 11.
3. `lang_009`: The answer is musically plausible but less diagnostic than desired because it omits the chord-tone relationship that makes both Cmaj7 and Am7 support the melody note E.

## Recommendation

Proceed to the harder reharmonization benchmark, but add prompting that explicitly asks the model to check melody notes against chord tones and avoid-note risks. For detailed harmonic spelling, manual review or RAG support is still advisable.
