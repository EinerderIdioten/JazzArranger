# Qwen3-8B Real Book Language Screening Test Report

## Executive Summary

Qwen3-8B was evaluated against the Real Book benchmark's language screening questions. This screening checks whether a small instruction model understands basic jazz harmony terminology before moving on to harder reharmonization tasks.

The model scored **17 / 20**, which is above the benchmark pass threshold of **14 / 20**. It is suitable for moving on to the harder reharmonization benchmark, but the run exposed a few predictable weaknesses around precise chord-extension spelling and avoid-note theory.

## Test Context

| Item | Value |
| --- | --- |
| Repository | `EinerderIdioten/jazzarranger` |
| Benchmark file | `realbook_benchmark/language_screening_questions.json` |
| Run directory | `test_logs/realbook_language_screening_qwen3_8b_20260729_144622` |
| Model | `qwen3-8b` |
| Model path on remote machine | `/root/autodl-tmp/models/Qwen3-8B` |
| Serving stack | vLLM OpenAI-compatible API |
| vLLM model name | `qwen3-8b` |
| API endpoint used during test | `http://127.0.0.1:18000/v1` via SSH tunnel |
| Run time | `2026-07-29T14:46:22+08:00` to `2026-07-29T14:46:39+08:00` |
| Question count | `10` |

## Prompting Method

Each question was sent as an independent OpenAI-compatible chat completion request. The benchmark recommended short natural-language answers, so the system prompt was:

```text
Answer each jazz harmony screening question briefly in 1-3 sentences. Do not generate full arrangements. Focus only on basic jazz harmony concepts. Do not show reasoning. /no_think
```

The model was called with:

- `temperature`: `0`
- `max_tokens`: `192`
- one user message containing exactly one screening question

`/no_think` was included to suppress Qwen3 reasoning output. The returned `reasoning_content` fields were empty apart from whitespace and were not scored.

## Scoring Method

The benchmark's own rubric was used:

| Score | Meaning |
| ---: | --- |
| `0` | Wrong or irrelevant |
| `1` | Partially correct |
| `2` | Correct and clear |

Maximum score: `20`

Pass threshold from the benchmark: `14 / 20`

## Result

**Final score: 17 / 20**

**Status: Pass**

The model knows the core vocabulary needed for a first-pass jazz-ballad reharmonization benchmark: major seventh chords, dominant resolution, ii-V-I motion, guide tones, rootless voicings, tritone substitution, and ballad pacing. The weak points were not broad conceptual failures; they were detail-level music-theory mistakes.

## Per-Question Results

| ID | Topic | Score | Human-readable assessment |
| --- | --- | ---: | --- |
| `lang_001` | Chord symbol meaning | 2 | Correctly identifies `Cmaj7` as C major with a major seventh and gives the chord tones `C, E, G, B`. |
| `lang_002` | Dominant function | 2 | Correctly describes `G7` in C major as dominant-function harmony resolving to C. It did not name guide tones, but the answer was clear enough for full credit. |
| `lang_003` | ii-V-I | 2 | Correctly gives `Dm7-G7-Cmaj7` and describes the motion as a common jazz progression resolving to tonic. |
| `lang_004` | Guide tones | 2 | Correctly names the 3rd and 7th as guide tones and describes their role in outlining chord changes. |
| `lang_005` | Rootless piano voicing | 2 | Correctly says the root is omitted and that upper structures/extensions carry the harmonic color, with appropriate bass-line context. |
| `lang_006` | Chord tensions | 1 | Correctly says 9, 11, and 13 add color/tension, but incorrectly explains them as intervals above the 7th rather than compound scale degrees above the root in chord context. |
| `lang_007` | Avoid notes | 1 | Correctly recognizes a clash with the major 3rd, but misstates the pitch spelling: natural 11 over Cmaj7 is `F`, and `#11` is `F#`, not `D`/`D#`. |
| `lang_008` | Tritone substitution | 2 | Correctly gives `Db7` as a tritone substitute for `G7` resolving to `Cmaj7` and explains the chromatic resolution. |
| `lang_009` | Melody compatibility | 1 | Correctly says both `Cmaj7` and `Am7` can support melody note `E`, but misses the diagnostic chord-tone explanation: `E` is the 3rd of `Cmaj7` and the 5th of `Am7`. |
| `lang_010` | Ballad pacing | 2 | Correctly explains that changing to a complex chord on every beat can clutter a slow ballad and reduce space for melody and phrasing. |

## What Went Well

The model reliably handled basic jazz language at the phrase/concept level:

- It understood common lead-sheet chord symbols.
- It understood dominant resolution and ii-V-I motion.
- It recognized guide tones and rootless voicing concepts.
- It identified tritone substitution correctly.
- It gave musically sensible answers about ballad pacing and avoiding over-dense harmony.

The responses were concise and generally aligned with the requested screening mode. Latencies were low for a local vLLM-served 8B model, roughly 1-3 seconds per question in this run.

## Issues Found

The model's main weakness is precise harmonic spelling:

1. **Chord extensions:** It treated `9`, `11`, and `13` as intervals measured above the 7th. That can lead to wrong voicing or analysis if carried into reharmonization.
2. **Avoid-note pitch names:** It understood that the natural 11th clashes with the 3rd on major seventh harmony, but it named the wrong notes for natural 11 and sharp 11.
3. **Melody-to-chord-tone explanation:** It recognized that `E` can work over both `Cmaj7` and `Am7`, but did not explicitly identify the chord-tone roles. This matters for a reharmonization benchmark where melody compatibility is central.

## Recommendation

Proceed to the harder reharmonization benchmark, but prompt the model to explicitly check:

- the melody note against each proposed chord,
- whether the melody is a chord tone, color tone, or avoid-note risk,
- the pitch spelling of extensions and alterations,
- whether each substitution preserves ballad pacing.

For evaluation, keep manual review in the loop. The model appears strong enough to produce plausible candidate reharmonizations, but not reliable enough to trust all detailed jazz-theory spelling without review or RAG support.

## Artifacts

This test folder contains:

- `metadata.json`: run metadata and vLLM model info
- `raw_responses.json`: full request/response records
- `responses.md`: readable question-by-question model answers
- `evaluation.md`: concise score summary
- `scores.json`: machine-readable score record
- `TEST_REPORT.md`: this human-readable report
