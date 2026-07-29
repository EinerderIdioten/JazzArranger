# Real Book-level Benchmark Cases

This folder contains small, original jazz-ballad reharmonization test cases for checking whether a model understands Real Book-level harmony and piano voicing concepts.

The cases are intentionally original fragments, not copied Real Book tunes. They are designed to test the same kind of reasoning needed for lead-sheet reharmonization:

- preserving melody compatibility
- enriching simple harmony
- using idiomatic jazz-ballad movement
- avoiding mechanical or overcomplicated substitutions
- explaining harmonic choices clearly

Current status: one sample case is included for review before expanding to ten cases.

## Files

- `sample_case_001.json`: one benchmark case using the repository's compact lead-sheet JSON style.
- `evaluation.md`: scoring rubric for manually evaluating the model response.
- `prompt_template.md`: prompt template for testing Qwen8B or another OpenAI-compatible model.
