# Evaluation Rubric for Sample Case 001

Use this rubric to manually evaluate the first Qwen8B response before writing an automated scorer.

Total: 100 points.

## 1. Melody compatibility: 25 points

The reharmonization should support the given melody notes, especially structural notes.

- 25: All structural melody notes are clearly supported as chord tones or idiomatic tensions.
- 15: Mostly compatible, with one questionable structural note.
- 5: Several melody notes clash with the chosen chords.
- 0: The harmony ignores the melody.

## 2. Jazz-ballad harmonic language: 20 points

The response should sound like plausible Real Book-level jazz ballad harmony.

- 20: Uses idiomatic major 9, minor 9, secondary dominant, sus dominant, or tritone substitute colors with good pacing.
- 12: Uses some jazz color but remains mostly plain or somewhat mechanical.
- 5: Adds complexity without musical purpose.
- 0: Does not sound like jazz harmony.

## 3. Functional voice-leading logic: 20 points

The response should show coherent harmonic motion.

- 20: Root motion and dominant resolution are clear and musically justified.
- 12: Some functional motion is present but not consistently explained.
- 5: Chords are decorative but functionally weak.
- 0: Chord sequence is random or contradictory.

## 4. Realistic piano-arrangement awareness: 15 points

Even if no voicings are required, the chord choices should be playable and suitable for piano ballad treatment.

- 15: Chords imply playable voicings and reasonable harmonic rhythm.
- 8: Mostly playable but too dense or too fast for a ballad.
- 3: Likely awkward or impractical.
- 0: Not suitable for piano arrangement.

## 5. Explanation quality: 20 points

The model should explain choices like a musician, not only list chord names.

- 20: Clearly explains melody support, function, resolution, and style.
- 12: Gives partial explanations but misses important decisions.
- 5: Generic explanation with little music-theory content.
- 0: No usable explanation.

## Suggested pass threshold

- 80-100: Strong Real Book-level understanding.
- 65-79: Usable but needs RAG or rule support.
- 50-64: Basic jazz terms recognized, but unreliable.
- Below 50: Not ready for this task.
