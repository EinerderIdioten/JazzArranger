# Prompt Template for Qwen8B Test

Use this prompt with `sample_case_001.json`.

```text
You are evaluating jazz-ballad reharmonization skill.

Task:
Given a simple lead-sheet phrase with melody and chord symbols, rewrite the harmony into richer Real Book-level jazz-ballad harmony for piano.

Rules:
1. Preserve the given melody.
2. Keep the style as slow jazz ballad.
3. Use idiomatic jazz harmony, but do not overcomplicate every beat.
4. Explain why each important substitution works.
5. Return strict JSON only.

Input case:
<PASTE sample_case_001.json HERE>

Return JSON with this shape:
{
  "reharmonized_harmony_stream": [
    [bar, beat, "ChordSymbol", duration_beats]
  ],
  "explanation": [
    "short explanation per important harmonic decision"
  ],
  "risk_notes": [
    "any melody clashes, avoid-note risks, or voicing cautions"
  ]
}
```
