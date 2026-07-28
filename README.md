# Jazz Harmony Arranger AI Backend

Phase 1 initiative for building a benchmark system that evaluates how well large language models understand and generate jazz harmony arrangements.

The project focuses first on testing raw model capability before adding retrieval-augmented generation, fine-tuning, or domain-specific training pipelines.

## Objective

Create a Python-based evaluation framework for measuring whether an LLM can produce musically valid jazz voicings under explicit theory constraints.

The benchmark will test tasks such as:

- generating rootless voicings
- generating Drop 2 and Drop 3 voicings
- respecting required guide tones
- avoiding incorrect or unstable tones
- matching melody-note constraints
- maintaining valid voice count and register behavior
- identifying common jazz harmony rules around 3rds, 7ths, tensions, and avoid notes

## Phase 1 Scope

Phase 1 is intentionally limited to repository setup, project planning, and a baseline benchmark design.

The first benchmark should answer a simple question:

> How well can a base LLM arrange jazz harmonies without RAG, fine-tuning, or specialized music-theory context?

This means the initial system will evaluate raw model intelligence against a small curated dataset of expert-labeled jazz voicing examples.

## Planned Repository Structure

```text
.
├── benchmark/
│   ├── dataset.json        # Starter benchmark cases
│   └── run_eval.py         # Baseline evaluation runner
├── src/
│   ├── models.py           # Pydantic data models
│   └── evaluator.py        # Music-theory scoring logic
├── tests/
│   └── ...                 # Unit tests for models and evaluator
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── ROADMAP.md              # Execution plan and milestones
```

No implementation has been added yet. This README captures the intended direction for discussion before code is created.

## Benchmark Concept

Each benchmark item will describe a jazz harmony task with structured inputs and expected expert outputs.

A test case is expected to include:

- test ID
- chord symbol
- melody note
- voicing constraints
- expert reference voicings
- required tones
- avoid tones

A model response is expected to return a structured voicing, such as scientific pitch notation:

```json
["F3", "B3", "Eb4", "Bb4"]
```

The evaluator will compare the generated voicing against the musical constraints and expert labels.

## Initial Scoring Dimensions

The baseline score will be calculated from four weighted categories:

| Category | Weight | Purpose |
| --- | ---: | --- |
| Voice count limit | 20% | Checks whether the model follows requested voicing size |
| Guide tones accuracy | 30% | Checks for essential 3rd and 7th chord tones |
| Avoid-note violations | 30% | Penalizes notes that conflict with the target chord quality |
| Voice-leading smoothness | 20% | Rewards compact, playable, musically coherent spacing |

The evaluator should also detect common harmony issues such as:

- missing 3rds or 7ths
- voice crossing
- invalid pitch names
- unwanted avoid notes, such as natural 11ths over major seventh chords
- voicings that do not satisfy the requested voicing type

## Starter Dataset Direction

The first dataset should contain five curated Level 1 to Level 2 examples focused on:

- rootless Type A voicings
- rootless Type B voicings
- Drop 2 voicings
- common ii–V–I harmony contexts
- major seventh, dominant seventh, minor seventh, and altered dominant colors

The starter dataset should remain small enough to manually inspect while still being diverse enough to reveal model weaknesses.

## Execution Roadmap

### Milestone 1: Base LLM Benchmark

Build the baseline benchmark system and evaluate raw model output on jazz voicing tasks.

Goals:

- define structured test cases
- create a small expert-labeled dataset
- implement automatic scoring
- run the first baseline evaluation against a selected model endpoint
- produce a readable score report

### Milestone 2: OMR and Text Pipeline

Process jazz-theory source material into structured data.

Goals:

- ingest text and notation examples
- extract chord symbols, voicing rules, and examples
- normalize musical data into structured representations
- prepare the foundation for a jazz harmony knowledge graph

### Milestone 3: Graph-RAG and Comparative Evaluation

Use the structured knowledge graph to improve model performance and compare results against the raw baseline.

Goals:

- implement retrieval over harmony rules and examples
- prompt models with retrieved context
- compare Base LLM versus Graph-RAG performance
- identify which harmony tasks improve with retrieval

## Proposed Dependencies

The expected Python dependencies for Phase 1 are:

- `music21` for music-theory parsing and pitch analysis
- `pydantic` for structured data models
- `pytest` for testing
- one or more LLM SDKs depending on the selected model endpoint

The exact LLM SDK should be chosen after deciding which model provider or open-source inference endpoint will be used first.

## Open Questions Before Implementation

Before writing code, we should confirm:

1. Which repository name should be used?
2. Should the repository be public or private?
3. Which model endpoint should be evaluated first?
4. Should model responses be requested as strict JSON only?
5. Should the first benchmark focus only on piano-style voicings, or also include guitar-friendly shapes?
6. What register range should the evaluator consider valid for each voicing type?
7. Should expert labels allow multiple valid answers per test case?

## Current Status

This project is at the planning stage.

The next step is to review this README and confirm the repository direction before creating code, adding the full project structure, or implementing the benchmark framework.

---

This README was drafted by an AI agent (OpenHands) on behalf of the repository owner.
