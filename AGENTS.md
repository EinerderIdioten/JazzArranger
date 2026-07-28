# Repository Notes

- Phase 1 scope is piano-only jazz ballad reharmonization: given simple harmony plus melody, produce richer jazz harmony.
- Phase 1 model comparison should keep the sample set and model fixed: Qwen2.5-7B-Instruct via vLLM as Base LLM vs Qwen2.5-7B-Instruct + RAG, using BAAI/bge-m3 embeddings and Qdrant.
- Real Book-style ingestion docs live in `docs/realbook_ingestion/`.
- Canonical lead-sheet transcription format is compact JSON streams with explicit `[bar, beat, value, duration_beats]` timing. Use `bar = 0` for pickup material and avoid packed positions like `2.66`.
- Keep raw transcription separate from derived analysis; do not put chord-tone roles such as `3` or `13` into raw melody streams unless an assisted benchmark mode explicitly needs them.
